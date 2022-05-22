import math
import sys

import PSID
import pickle

import numpy as np

sys.path.append("..")
from utils.csp_example import *
from utils.psid_standalone_original import *
from utils.features_utils import power_average



def PSID_features_nostim(epochs, ica_model, include_stim = True):

    #stim labels
    stim_idx = epochs.info['ch_names'].index('stim')
    stim_data = np.nanmean(epochs.get_data()[:, stim_idx, :], axis=1)
    print(stim_data)


    #block labels
    blk_idx = epochs.info['ch_names'].index('blk_idx')
    blk_data = np.nanmean(epochs.get_data()[:, blk_idx, :], axis=1)

    #EEG data
    pick_ch_y = [ch for ch in epochs.info['ch_names'] if ch in ica_model.info['ch_names']]
    epochs_ica = project_ica(epochs.copy().pick_types(eeg=True).pick_channels(pick_ch_y), ica_model)
    data_Y = split_to_psid_model(epochs.copy().pick_types(eeg=True))#epochs_ica)
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

    Y_stack = np.empty((Y.shape[1], Y.shape[2]))
    Z_stack = np.empty((Z.shape[1], Z.shape[2]))

    nb_train_epochs = math.floor(nb_epochs_*0.8)

    for ei in range(0, nb_train_epochs,  step_epochs):
        print(ei)
        tstart = time.time()
        Ye, Ze = np.asarray(Y[ei]), np.asarray(Z[ei])
        #print("Ye shape : ", Ye.shape, "Ze shape : ", Ze.shape)
        Y_stack = np.concatenate([Y_stack, Ye], axis=1)
        Z_stack = np.concatenate([Z_stack, Ze], axis=1)


    print("Y_stack shape : ", Y_stack.shape, "Z_stack shape : ", Z_stack.shape)


    idSys = PSID.PSID(np.transpose(Y_stack), np.transpose(Z_stack), nx, n1, i)

    for ei in range(nb_train_epochs  + 1, nb_epochs_, step_epochs):
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

            y.append(stim_data[ei])
            blocks_idx.append(blk_data[ei])

        print(f"Finished after {time.time() - tstart} seconds")

    # make y as binary variable
    y = [1 if yi > 0.15 else 0 for yi in y]

    X_mean = np.array(X_mean)
    X_psd_m = np.array(X_psd_m)
    y = np.array(y)

    return X_psd_m, y, blocks_idx

exps = [str(n) for n in range(1, 3)]

for exp in exps:
    epochs = mne.read_epochs("data/VP" + exp + "_epo.fif")
    ica_model = mne.preprocessing.read_ica("data/VP" + exp + "_ica.fif")
    X, y, blocks_idx = PSID_features_nostim(epochs, ica_model, include_stim=True)

    with open('results/psid_' + exp + '_stimtest.pickle', 'wb') as handle:
        pickle.dump((X, y, blocks_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)