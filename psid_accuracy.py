import argparse
import os
import time
import math
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
import PSID

from sklearn.model_selection import LeavePGroupsOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold

import xgboost as xgb

from utils.classif_utils import MyLeaveNOut
from utils.taper_epochs import taper_epoch_bounds, concat_epochs_data
from utils.psid_standalone_original import split_to_psid_model, normalize_and_fill_nan
from utils.features_utils import power_average
from utils.csp_example import project_ica

cv = KFold(n_splits=5, shuffle=True)
scorer = make_scorer(accuracy_score)
xgb_model = xgb.XGBClassifier(max_depth=5,
                          n_estimators=10,
                          n_jobs=3,
                          eval_metric="logloss",
                          use_label_encoder=False)
lda_model = LinearDiscriminantAnalysis("eigen",shrinkage="auto")

def signal_to_features(epochs, 
                        ica_model,
                        freq_buckets,
                        behav_var="all",
                        nx = 8,
                        n1 = 8,
                        i_psid = 5
                        ):

    epochs_data = epochs.get_data()
    sfreq = epochs.info["sfreq"]

    
    features_dicts = []

    # get block and stim data, define crossval splits
    

    blk_idx = epochs.info['ch_names'].index('blk_idx')
    blk_data = np.nanmean(epochs_data[:, blk_idx, :], axis=1)
    stim_idx = epochs.info['ch_names'].index('stim')
    stim_data = np.nanmean(epochs_data[:, stim_idx, :], axis=1)

    leaveNout = MyLeaveNOut()
    # LeavePGroupsOut(2)
    splits = leaveNout.split(epochs_data, stim_data, blk_data)

    #print("NB SPLITS : ", len(splits))

    # keep only channels of interest

    if behav_var == "stim":
        epochs = epochs.copy().drop_channels(['dist_t_n',
                                            'pos_t_x',
                                            'pos_t_y',

                                            'speed_t_x',
                                            'speed_t_y',
                                            'accel_t_x',
                                            'accel_t_y',
                                            'jerk_t_x',
                                            'jerk_t_y'])
    elif behav_var == "no_stim":
        epochs = epochs.copy().drop_channels(['stim'])

    pick_ch_y = [ch for ch in epochs.info['ch_names'] if ch in ica_model.info['ch_names']]

    # get EEG data (ICA)
    epochs_ica = project_ica(epochs.copy().pick_types(eeg=True).pick_channels(pick_ch_y), ica_model)
    epochs_data_normalized = normalize_and_fill_nan(split_to_psid_model(epochs_ica))
    Y = epochs_data_normalized["Y"]

    # get behavioral data
    epochs_data_normalized = normalize_and_fill_nan(split_to_psid_model(epochs))
    Z = epochs_data_normalized["Z"]
    

    for isplit, (ix_train, ix_test) in enumerate(splits):

        print("=" * 80)
        print(f"Processing split {isplit + 1}")

        # train PSID on train data

        Y_train_stack = concat_epochs_data(Y[ix_train])
        Z_train_stack = concat_epochs_data(Z[ix_train])

        print(Y_train_stack.shape, Z_train_stack.shape)

        idSys = PSID.PSID(np.transpose(Y_train_stack), np.transpose(Z_train_stack), nx, n1, i_psid)

        # get predicted Y and Z 
        
        Z_true = Z[ix_test]
        Y_true = Y[ix_test]

        Z_pred = np.asarray([np.transpose(idSys.predict(np.transpose(Y[ei]))[0]) for ei in ix_test])
        Y_pred = np.asarray([np.transpose(idSys.predict(np.transpose(Y[ei]))[1]) for ei in ix_test])
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # get predicted X and extract its features using latent_to_features

        for ei in ix_train:
            xPred = np.transpose(idSys.predict(np.transpose(Y[ei]))[2])
            xFeatures = latent_to_features(xPred,freq_buckets, sfreq)

            X_train.extend(xFeatures)
            y_train.extend([stim_data[ei]]*len(xFeatures))

        for ei in ix_test:
            xPred = np.transpose(idSys.predict(np.transpose(Y[ei]))[2])
            xFeatures = latent_to_features(xPred,freq_buckets, sfreq)

            X_test.extend(xFeatures)
            y_test.extend([stim_data[ei]]*len(xFeatures))


        y_train = np.asarray([1 if yi > 0.15 else 0 for yi in y_train])
        X_train = np.array(X_train)
        y_test = np.asarray([1 if yi > 0.15 else 0 for yi in y_test])
        X_test = np.array(X_test)

        features_dicts.append(
            {
                "Z_true" : Z_true,
                "Z_pred" : Z_pred,
                "Y_true" : Y_true,
                "Y_pred" : Y_pred,

                "X_train" : X_train,
                "y_train" : y_train,
                "X_test" : X_test,
                "y_test" : y_test,
            })

        with open('results/psid_features_dicts.pickle', 'wb') as handle:
            pickle.dump(features_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    return features_dicts


def latent_to_features(xPred,freq_buckets,sfreq=300):

    nb_timesteps=len(xPred[0])
    nb_segments = 3
    segment_len = int(nb_timesteps / 5)

    X_psd_m = []
    for s in range(nb_segments):
        signal = np.array(xPred[:, s * segment_len:(s + 1) * segment_len])

        # psd multitaper over segment
        fft_signal = mne.time_frequency.psd_array_multitaper(signal, sfreq=sfreq, fmin=5, fmax=132, verbose=None)

        X_psd_m.append(np.log(np.ndarray.flatten(power_average(fft_signal, freq_buckets))))
    X_psd_m = np.array(X_psd_m)
    
    return X_psd_m

def psid_metrics(result_dicts):

    dist = []
    corr = []
    lda_scores = []
    xgb_scores = []
    xgb_fi = []
    lda_scores_test_only = []

    for split_idx, res in enumerate(result_dicts):

        Z_true = res["Z_true"]
        Z_pred = res["Z_pred"]
        Y_true = res["Y_true"]
        Y_pred = res["Y_pred"]
        X_train = res["X_train"]
        y_train = res["y_train"]
        X_test = res["X_test"]
        y_test = res["y_test"]
        """


        Z_true_concat = np.hstack(Z_true) # (n_channels, n_epochs*n_times)
        Z_pred_concat = np.hstack(Z_pred) # (n_channels, n_epochs*n_times)


        Z_kendall = np.asarray([stats.kendalltau(Zt, Zp)[0] for Zt, Zp in zip(Z_true_concat, Z_pred_concat)])
        Z_dist = np.linalg.norm(Z_true_concat-Z_pred_concat, axis=1)
        
        corr.append(Z_kendall[-1])
        dist.append(Z_dist[-1])

        
        """
        # LDA CLASSIF 
        lda_model.fit(X_train,y_train)
        lda_scores.append(scorer(lda_model,X_test,y_test))

        # XGB CLASSIF + FEATURE IMPORTANCE 
        xgb_model.fit(X_train,y_train)
        xgb_scores.append(scorer(xgb_model,X_test,y_test))
        xgb_fi.append(xgb_model.feature_importances_.reshape((7, nx)))
        
        # LDA CLASSIF WHEN USING TEST SET ONLY
        classif_splits = list(cv.split(X_test, y_test))
        for isplit, (classif_ix_train, classif_ix_test) in enumerate(classif_splits):
            lda_model.fit(X_test[classif_ix_train], y_test[classif_ix_train])

            lda_scores_test_only.append(scorer(lda_model, X_test[classif_ix_test], y_test[classif_ix_test]))
        

    return corr, dist, lda_scores, xgb_scores, xgb_fi, lda_scores_test_only


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PSID_acc')

    parser.add_argument('--exp_idx', type=int, default=1, help='index of the experiment to run')
    parser.add_argument('--data_dir', type=str, default='data directory')
    parser.add_argument('--nx', type=int, default=15, help='dimension of the latent space')
    parser.add_argument('--n1', type=int, default=15, help='dimension of the behavior-related latent space')

    args = parser.parse_args()

    data_dir = args.data_dir
    exp_idx = args.exp_idx
    nx = args.nx
    n1 = args.n1

    corr_list, dist_list, xgb_scores_list, lda_scores_list, lda_scores_test_only_list, xgb_fi_list = [], [], [], [], [], []


    freq_ranges = [(5, 8), (8, 13), (13, 30)]
    freq_nbs = [1, 3, 3]

    fb = []
    for i_, f in enumerate(freq_ranges):
        points = np.linspace(f[0], f[1], num=freq_nbs[i_] + 1)
        fb += [(points[i], points[i + 1]) for i in range(len(points) - 1)]

    time_list = []
    
    # grid search on the i_psid_ratio parameter

    i_psid_ratio_list = [0.01,0.02,0.05]
    for i_psid_ratio in i_psid_ratio_list:

        """

        epochs = mne.read_epochs(os.path.join(data_dir,"VP" + str(exp_idx) + "_epo.fif"))
        sfreq = epochs.info["sfreq"]
        ica_model = mne.preprocessing.read_ica(os.path.join(data_dir,"VP" + str(exp_idx) + "_ica.fif"))

        i_psid = math.floor(sfreq*i_psid_ratio)
        print("I PSID : ", i_psid)

        tstart = time.time()

        features_dicts = signal_to_features(epochs, 
                                            ica_model, 
                                            freq_buckets=fb,
                                            behav_var="all",
                                            nx=nx,
                                            n1=n1,
                                            i_psid=i_psid
                                            )
        time_list.append(time.time() - tstart)
        

        with open('results/psid_features_' + str(exp_idx) + "_i_" + str(i_psid) + '.pickle', 'wb') as handle:
            pickle.dump(features_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """
        with open('results/psid_features_result_dicts.pickle', 'rb') as handle:
            features_dicts = pickle.load(handle)#, protocol=pickle.HIGHEST_PROTOCOL)



        corr, dist, lda_scores, xgb_scores, xgb_fi, lda_scores_test_only = psid_metrics(features_dicts)
        corr_list.append(corr)
        dist_list.append(dist)
        lda_scores_list.append(lda_scores)
        xgb_scores_list.append(xgb_scores)
        xgb_fi_list.append(xgb_fi)
        lda_scores_test_only_list.append(lda_scores_test_only)


    label_list = i_psid_ratio_list

    plt.scatter(x=range(len(time_list)),y=time_list )

    plt.boxplot(corr_list, labels=label_list)
    plt.title("Kendall's Tau between original and predicted behavioral data")
    plt.show()
    plt.boxplot(dist_list, labels=label_list)
    plt.title("Distance between original and predicted behavioral data")
    plt.show()

    plt.boxplot(lda_scores_list, labels=label_list)
    plt.title("lda classif accuracy")
    plt.show()
    plt.boxplot(lda_scores_test_only_list, labels=label_list)
    plt.title("lda classif accuracy (on test data only)")
    plt.show()

    plt.boxplot(xgb_scores_list, labels=label_list)
    plt.title("xgb classif accuracy")
    plt.show()

    for i_psid_idx, xgb_fi in enumerate(xgb_fi_list):
        fig, axs = plt.subplots(1,len(xgb_fi))
        print(str(i_psid_ratio_list[i_psid_idx]))
        plt.title("xgb feature importance for i psid ratio = " + str(i_psid_ratio_list[i_psid_idx]))
        for i,x in enumerate(xgb_fi):
            axs[i].imshow(x,vmin=0, vmax=1)

        plt.show()





