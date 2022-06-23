import sys
import PSID
import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
from utils.classif_utils import MyLeaveNOut
from utils.taper_epochs import taper_epoch_bounds, concat_epochs_data
from utils.psid_standalone_original import split_to_psid_model, normalize_and_fill_nan
from utils.features_utils import power_average
import scipy.stats as stats
import time
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold


scorer = make_scorer(accuracy_score)
xgb_model = xgb.XGBClassifier(max_depth=5,
                          n_estimators=10,
                          n_jobs=3,
                          eval_metric="logloss",
                          use_label_encoder=False)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_model = LinearDiscriminantAnalysis("eigen",shrinkage="auto")

def data_reconstruction(epochs, ica_model,
                  behav_var="all", #"stim", "no_stim",
                  n1 = 8,
                  i_psid = 5
                  ):

    epochs_data = epochs.get_data()

    nx = n1
    result_dicts = []

    # get block and stim data, define crossval splits
    

    blk_idx = epochs.info['ch_names'].index('blk_idx')
    blk_data = np.nanmean(epochs_data[:, blk_idx, :], axis=1)
    stim_idx = epochs.info['ch_names'].index('stim')
    stim_data = np.nanmean(epochs_data[:, stim_idx, :], axis=1)

    leaveNout = MyLeaveNOut()
    splits = leaveNout.split(epochs_data, stim_data, blk_data)

    print("NB SPLITS : ", len(splits))

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

    # get epochs data

    epochs_data_normalized = normalize_and_fill_nan(split_to_psid_model(epochs))

    Y = epochs_data_normalized["Y"]
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
        
        #Z_true = Z[ix_test]
        #Y_true = Y[ix_test]

        #Z_pred = np.asarray([np.transpose(idSys.predict(np.transpose(Y[ei]))[0]) for ei in ix_test])
        #Y_pred = np.asarray([np.transpose(idSys.predict(np.transpose(Y[ei]))[1]) for ei in ix_test])
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # get predicted X and extract its features using latent_to_features

        for ei in ix_train:
            xPred = np.transpose(idSys.predict(np.transpose(Y[ei]))[2])
            xFeatures = latent_to_features(xPred)

            X_train.extend(xFeatures)
            y_train.extend([stim_data[ei]]*len(xFeatures))

        for ei in ix_test:
            xPred = np.transpose(idSys.predict(np.transpose(Y[ei]))[2])
            xFeatures = latent_to_features(xPred)

            X_test.extend(xFeatures)
            y_test.extend([stim_data[ei]]*len(xFeatures))


        y_train = np.asarray([1 if yi > 0.15 else 0 for yi in y_train])
        X_train = np.array(X_train)
        y_test = np.asarray([1 if yi > 0.15 else 0 for yi in y_test])
        X_test = np.array(X_test)

        result_dicts.append(
            {
                #"Z_true" : Z_true,
                #"Z_pred" : Z_pred,
                #"Y_true" : Y_true,
                #"Y_pred" : Y_pred,

                "X_train" : X_train,
                "y_train" : y_train,
                "X_test" : X_test,
                "y_test" : y_test,
            })

    return result_dicts


def latent_to_features(xPred):

    nb_timesteps=len(xPred[0])
    nb_segments = 3
    segment_len = int(nb_timesteps / 5)
    sfreq = 200  # sampling frequency

    freq_ranges = [(5, 8), (8, 13), (13, 30)]
    freq_nbs = [1, 3, 3]

    all_freq_ranges = []
    for i_, f in enumerate(freq_ranges):
        points = np.linspace(f[0], f[1], num=freq_nbs[i_] + 1)
        all_freq_ranges += [(points[i_], points[i_ + 1]) for i in range(len(points) - 1)]

    X_psd_m = []
    for s in range(nb_segments):
        signal = np.array(xPred[:, s * segment_len:(s + 1) * segment_len])

        # psd multitaper over segment
        fft_signal = mne.time_frequency.psd_array_multitaper(signal, sfreq=sfreq, fmin=5, fmax=30, verbose=None)
        X_psd_m.append(np.ndarray.flatten(power_average(fft_signal, all_freq_ranges)))
    X_psd_m = np.array(X_psd_m)
    
    return X_psd_m

def psid_metrics(result_dicts, n1):

    xgb_fi = []
    xgb_scores = []
    lda_scores = []

    for split_idx, res in enumerate(result_dicts):

        
        """
        Z_true = res["Z_true"]
        Z_pred = res["Z_pred"]
        Y_true = res["Y_true"]
        Y_pred = res["Y_pred"]
        """
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

        xgb_model.fit(X_train,y_train)
        xgb_scores.append(scorer(xgb_model,X_test,y_test))
        xgb_fi.append(xgb_model.feature_importances_.reshape((7, n1)))

        lda_model.fit(X_train,y_train)
        lda_scores.append(scorer(lda_model,X_test,y_test))

        """

        classif_splits = list(cv.split(X_test, y_test))
        for isplit, (classif_ix_train, classif_ix_test) in enumerate(classif_splits):
            model.fit(X_test[classif_ix_train], y_test[classif_ix_train])

            scores_test_only.append(scorer(model, X_test[classif_ix_test], y_test[classif_ix_test]))
        """

    print(xgb_scores,np.mean(xgb_scores))
    print(lda_scores,np.mean(lda_scores))
    return xgb_scores, xgb_fi, lda_scores


exps = [str(n) for n in [1]]#2,3,4,7,8]]
behav_var_list = ["all"]
behav_var = "all"
n1_list = [2,5,10,15,20,30]
n1 = 15
xgb_scores_list, xgb_fi_list, lda_scores_list, times_list = [], [], [], []


for exp in exps:
    for n1 in n1_list:
        print("n1:",n1)
        
        tstart = time.time()
        
        
        """
        epochs = mne.read_epochs("data/VP" + exp + "_epo.fif")
        ica_model = mne.preprocessing.read_ica("data/VP" + exp + "_ica.fif")

        
        result_dicts = data_reconstruction(epochs, 
                                            ica_model, 
                                            behav_var=behav_var,
                                            n1=n1,
                                            i_psid=15
                                            )


        
        with open('results/psid_features_' + exp + "_" + behav_var + "_n1_" + str(n1) + '_hpo.pickle', 'wb') as handle:
            pickle.dump(result_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        """
        result_dicts = pickle.load(open('results/psid_features_' + exp + "_" + behav_var + "_n1_" + str(n1) + '_hpo.pickle','rb'))
        
        xgb_scores, xgb_fi, lda_scores = psid_metrics(result_dicts, n1)
        xgb_scores_list.append(xgb_scores)
        xgb_fi_list.append(xgb_fi)
        lda_scores_list.append(lda_scores)
        times_list.append(time.time()-tstart)


    
    fig, axs = plt.subplots(1,2)
    axs[0].boxplot(xgb_scores_list, labels=n1_list)
    axs[0].set_title("xgb scores for different n1")
    axs[0].set_ylim(0,1)
    axs[1].boxplot(lda_scores_list, labels=n1_list)
    axs[1].set_title("lda scores for different n1")
    axs[1].set_ylim(0,1)
    plt.show()
    plt.plot(n1_list, times_list)
    plt.title("time for different n1")
    plt.show()
    for xgb_fi in xgb_fi_list:
        fig, axs = plt.subplots(1,len(xgb_fi))
        for i,x in enumerate(xgb_fi):
            axs[i].imshow(x,vmin=0, vmax=1)
        plt.show()

    xgb_scores_list = []
    xgb_fi_list = []
    lda_scores_list = []
    times_list = []


    


