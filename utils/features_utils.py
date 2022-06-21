from utils.csp_example import *
from utils.classif_utils import MyLeaveNOut
import matplotlib.pyplot as plt
import PSID
import sklearn
from utils.psid_standalone_original import *
from sklearn.model_selection import LeavePGroupsOut

def power_average(fft_signal, all_freq_ranges):
    powers = []
    for fr in all_freq_ranges:
        a = [fft_signal[0][:, i] for i in range(fft_signal[1].shape[0]) if fr[0] <= fft_signal[1][i] < fr[1]]
        powers.append(np.mean(a, axis=0))

    return np.array(powers)


### get (PSID + PSD) and (PSID + average) features, by time segment


def CSP_LDA_features(epochs, ica_model, tf=[25.38, 30.38]):

    # Get stim and block data
    stim_idx = epochs.info['ch_names'].index("stim")
    blk_idx = epochs.info['ch_names'].index('blk_idx')

    stim_data = np.nan_to_num(epochs.get_data()[:, stim_idx, :])
    blk_data = np.nanmean(epochs.get_data()[:, blk_idx, :], axis=1)

    pick_ch = [ch for ch in epochs.info['ch_names'] if ch in ica_model.info['ch_names']]
    epochs_eeg = epochs.copy().pick_types(eeg=True).pick_channels(pick_ch)


    mini_epochs_list, _ = create_subepochs(epochs_eeg, target_frequency=tf,
                                           reject_frequency=[1, 30],
                                           subepo_crop_tmin=0)

    mini_epochs_list = [e.load_data() for e in mini_epochs_list if e != []]

    non_empty_idx = [idx for idx, e in enumerate(mini_epochs_list) if e != []]

    ica_mini_epochs = [project_ica(e, ica_model) for e in mini_epochs_list]

    y = []
    for ei in non_empty_idx:
        y.append(np.mean(stim_data[ei]))

    y = [1 if yi > 0.15 else 0 for yi in y]
    y = np.asarray(y)

    X = np.asarray(ica_mini_epochs, dtype='object')

    return X, y, blk_data[non_empty_idx]



def PSID_features(epochs, ica_model, include_stim = True):

    #stim labels
    stim_idx = epochs.info['ch_names'].index('stim')
    stim_data = np.nanmean(epochs.get_data()[:, stim_idx, :], axis=1)


    #block labels
    blk_idx = epochs.info['ch_names'].index('blk_idx')
    blk_data = np.nanmean(epochs.get_data()[:, blk_idx, :], axis=1)

    #EEG data
    pick_ch_y = [ch for ch in epochs.info['ch_names'] if ch in ica_model.info['ch_names']]
    epochs_ica = project_ica(epochs.copy().pick_types(eeg=True).pick_channels(pick_ch_y), ica_model)
    data_Y = split_to_psid_model(epochs_ica)
    data_Y = normalize_and_fill_nan(data_Y)
    Y = data_Y['Y']

    #Behavioral data
    if include_stim:
        pick_ch_z = epochs.info['ch_names']
    else:
        pick_ch_z = [ch for ch in epochs.info['ch_names'] if ch != "stim"]

    data_Z = split_to_psid_model(epochs.copy().pick_channels(pick_ch_z))
    data_Z = normalize_and_fill_nan(data_Z)
    Z = data_Z['Z']

    #parameters of the feature extraction
    #PSID
    i = 5
    n1 = 20
    nx = n1

    #frequency split
    freq_ranges = [(5, 8), (8, 13), (13, 30)]
    freq_nbs = [1, 3, 3]

    all_freq_ranges = []
    for i_, f in enumerate(freq_ranges):
        points = np.linspace(f[0], f[1], num=freq_nbs[i_] + 1)
        all_freq_ranges += [(points[i_], points[i_ + 1]) for i in range(len(points) - 1)]
    print(all_freq_ranges)

    nb_segments = 3
    nb_epochs_ = Y.shape[0]
    step_epochs = 1
    sfreq = 200  # sampling frequency

    X_mean = []
    X_psd_m = []
    y = []
    blocks_idx = []
    Cz_list = []

    for ei in range(0, nb_epochs_, step_epochs):
        print(ei)
        tstart = time.time()
        Ye, Ze = Y[ei], Z[ei]
        print(Ye.shape,Ze.shape)

        idSys = PSID.PSID(np.transpose(Ye), np.transpose(Ze), nx, n1, i)
        zPred, yPred, xPred = idSys.predict(np.transpose(Ye))
        Cz_list.append(idSys.Cz)

        nb_timesteps = len(np.transpose(xPred)[0])
        segment_len = int(nb_timesteps / 5)

        # for all segments of interest in the epoch
        for s in range(nb_segments):
            signal = np.array(np.transpose(xPred)[:, s * segment_len:(s + 1) * segment_len])

            # mean signal over segment
            X_mean.append(np.mean(signal, axis=1))

            # psd multitaper over segment
            fft_signal = mne.time_frequency.psd_array_multitaper(signal, sfreq=sfreq, fmin=5, fmax=30, verbose=None)
            X_psd_m.append(np.ndarray.flatten(power_average(fft_signal, all_freq_ranges)))

            y.append(stim_data[ei])
            blocks_idx.append(blk_data[ei])

        print(f"Finished after {time.time() - tstart} seconds")

    # make y as binary variable
    y = [1 if yi > 0.15 else 0 for yi in y]

    X_mean = np.array(X_mean)
    X_psd_m = np.array(X_psd_m)
    y = np.array(y)

    return {"X" : X_psd_m,
            "y" : y,
            "blocks_idx" : blocks_idx,
            "Cz_list" : Cz_list }




def random_crossval(X, y, model, metric = auc_scoring, do_plot=False):

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    splits = list(cv.split(X, y))
    scorer = make_scorer(metric)
    scores = []
    #print(splits)

    for isplit, (ix_train, ix_test) in enumerate(splits):
        print("=" * 80)
        print(f"Processing split {isplit + 1}")

        model.fit(X[ix_train], y[ix_train])

        scores.append(scorer(model, X[ix_test], y[ix_test]))

    print(f"Scores: {scores} - {np.mean(scores):.2%}")
    if do_plot:
        plt.boxplot(scores)
        plt.show()

    return scores

def block_crossval(X, y, model_class, blk, metric = roc_auc_score, do_plot=False, do_feature_imp=False,leaveP=True):

    if leaveP:
        leaveNout = LeavePGroupsOut(1)
    else:
        leaveNout = MyLeaveNOut()
    splits = leaveNout.split(X, y, blk)

    

    scorer = make_scorer(metric)
    scores = []
    feature_imp = []

    for isplit, (ix_train, ix_test) in enumerate(splits):
        print("=" * 80)
        print(f"Processing split {isplit + 1}")
        model = sklearn.base.clone(model_class)
        model.fit(X[ix_train], y[ix_train])
        scores.append(scorer(model, X[ix_test], y[ix_test]))
        if do_feature_imp:
            feature_imp.append(model.feature_importances_)

    print(f"Scores: {scores} - {np.mean(scores):.2%}")
    if do_plot:
        plt.boxplot(scores)
        plt.show()

    if do_feature_imp:
        return scores, feature_imp
    else:
        return scores
"""
    model = xgb.XGBClassifier(max_depth=5,
                              n_estimators=10,
                              n_jobs=3,
                              eval_metric="logloss",
                              use_label_encoder=False)


    cv = StratifiedKFold(n_splits=5, shuffle=True)


    splits = list(cv.split(X_psd_m, y))
    scorer = make_scorer(auc_scoring)
    scores = []


    for isplit, (ix_train, ix_test) in enumerate(splits):

        print("=" * 80)
        print(f"Processing split {isplit + 1}")

        model.fit(X_psd_m[ix_train], y[ix_train])
        scores.append(scorer(model, X_psd_m[ix_test], y[ix_test]))

    plt.boxplot(scores)
    plt.show()
    print(f"Scores: {scores} - {np.mean(scores):.2%}")


    leaveNout = MyLeaveNOut()

    splits = leaveNout.split(X_psd_m, y, blocks_idx)

    scores = []
    feature_imp = []

    for isplit, (ix_train, ix_test) in enumerate(splits):
        model.fit(X_psd_m[ix_train], y[ix_train])
        feature_imp.append(model.feature_importances_.reshape((7, 20)))
        s = model.score(X_psd_m[ix_test], y[ix_test])
        scores.append(s)

    feature_imp = np.asarray(feature_imp)
    plt.boxplot(scores)
    plt.show()
    print(f"Scores: {scores} - {np.mean(scores):.2%}")
    
    
def psid_param_eval(Y,
                    Z,
                    blk_data,
                    i=20,
                    n1=8,
                    nb_epochs_=132,
                    step_epochs=1,
                    freq_ranges=[(5, 8), (8, 13), (13, 30)],
                    freq_nbs=[1, 3, 3]):
    # frequency buckets

    all_freq_ranges = []
    for i_, f in enumerate(freq_ranges):
        points = np.linspace(f[0], f[1], num=freq_nbs[i_] + 1)
        all_freq_ranges += [(points[i_], points[i_ + 1]) for i in range(len(points) - 1)]
    print(all_freq_ranges)

    nb_timesteps = Y.shape[2]  # nb timesteps per epoch (2301)

    # number/length of segments for
    nb_segments = 3
    segment_len = int(nb_timesteps / 5)
    ###utils for splitting psd results into frequency buckets

    sfreq = 200  # sampling frequency

    nx = n1

    X_mean = []
    X_psd_m = []
    y = []
    blocks_idx = []

    for ei in range(0, nb_epochs_, step_epochs):
        print(ei)
        tstart = time.time()
        Ye, Ze = Y[ei], Z[ei]

        idSys = PSID.PSID(np.transpose(Ye), np.transpose(Ze), nx, n1, i)
        zPred, yPred, xPred = idSys.predict(np.transpose(Ye))

        nb_timesteps = len(np.transpose(xPred)[0])
        segment_len = int(nb_timesteps / 5)

        # for all segments of interest in the epoch
        for s in range(nb_segments):
            signal = np.array(np.transpose(xPred)[:, s * segment_len:(s + 1) * segment_len])

            # mean signal over segment
            X_mean.append(np.mean(signal, axis=1))

            # psd multitaper over segment
            fft_signal = mne.time_frequency.psd_array_multitaper(signal, sfreq=sfreq, fmin=5, fmax=30, verbose=None)
            X_psd_m.append(np.ndarray.flatten(power_average(fft_signal, all_freq_ranges)))

            #print(power_average(fft_signal, all_freq_ranges).shape)

            y.append(np.mean(Ze[-1, :]))
            blocks_idx.append(blk_data[ei])

        print(f"Finished after {time.time() - tstart} seconds")

    # make y as binary variable
    y = [1 if yi > 0.15 else 0 for yi in y]

    X_mean = np.array(X_mean)
    X_psd_m = np.array(X_psd_m)
    y = np.array(y)

    return X_mean, X_psd_m, y, blocks_idx
"""