#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220124
#
# A standalone implementation of the PSID step
#
# ---> PSID Paper
# Sani, O.G., Abbaspourazad, H., Wong, Y.T. et al. Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification. Nat Neurosci 24, 140–149 (2021). https://doi.org/10.1038/s41593-020-00733-0          # noqa


# The PSID model is based of a statespace formulated as follows:
#     x_(k+1) = Ax_(k) + w_(k)
#     y_(k) = C_(y)x_(k) + nu_(k)
#     z_(k) = C_(z)x_(k) + e_(k)

#     with w_(k) and nu_(k) being zero mean white noise independent of x_(k)
#     and e_(k) being a general noise term for behavior not covered by
#     C_(z)x_(k)

#     x <- hidden neural activity
#     y <- measured neural activity
#     z <- behavior neural activity

#     It then assumes that x_(k) can be split in relevant and non-relevant sub-
#     spaces for the behavior:
#         x_(k) = [x^r_(k), x^nr_(k)]^T

import pdb
import mne
import time
import scipy

import numpy as np


def split_to_psid_model(epo):
    """Split channels to neural y_(k) and behavioral z_(k) subspaces"""

    # EEG channels
    y_picks = mne.pick_types(epo.info, eeg=True)
    eeg_ch_names = [epo.info['ch_names'][i]
                    for i in range(epo.info['nchan']) if i in y_picks]

    # Behavioral channels
    # Note: Those where synthetically added during preprocessing
    behav_stems = ['dist_t', 'pos_t', 'speed_t', 'accel_t', 'jerk_t', 'stim']
    z_picks = [i for i in mne.pick_types(epo.info, misc=True)
               if any(pf in epo.info['ch_names'][i]
                      for pf in behav_stems)]
    behav_ch_names = [epo.info['ch_names'][i]
                      for i in range(epo.info['nchan']) if i in z_picks]

    # NOTE: Indices are <n_epochs> x <n_channels> x <n_times>
    Y = epo.get_data()[:, y_picks, :]
    Z = epo.get_data()[:, z_picks, :]

    # NOTE: each of the epochs has leading and trailing nan for the behavioral
    # channels, e.g. epochs might start at -1, but behavioral data is only
    # available for t > 0.
    nnm = np.all(np.all(~np.isnan(Z), axis=0), axis=0)

    data = {
        'Y': Y,
        'Z': Z,
        'times': epo.times[nnm],
        'Z_ch_names': behav_ch_names,
        'Y_ch_names': eeg_ch_names
    }

    return data


def normalize_and_fill_nan(data):
    Y = data['Y']
    Z = data['Z']

    means_y = np.asarray([Y.mean(axis=2).T] * Y.shape[2]).T
    means_z = np.asarray([np.nan_to_num(Z).mean(axis=2).T] * Z.shape[2]).T

    std_y = np.asarray([Y.std(axis=2).T] * Y.shape[2]).T
    std_z = np.asarray([np.nan_to_num(Z).std(axis=2).T] * Z.shape[2]).T

    Y = np.nan_to_num((Y.data - means_y) / std_y)
    Z = np.nan_to_num((Z.data - means_z) / std_z)

    data.update(
        {
            'Z': Z,
            'Y': Y,
            'means_z': means_z,
            'means_y': means_y,
            'std_z': std_z,
            'std_y': std_y
        }
    )

    return data


def compute_psid_models(data):

    # TODO: Optimize this loop
    # --> over each admissible time step which provides sufficient future
    # and history and accross all epochs
    # Idea for the epochs -> orthogonal subspaces => not feasible (for one
    # epoch Yp ~ 150mb, not enough memory for concatenation as data is dense)
    Y = data['Y']
    Z = data['Z']

    # Using nomenclature from the paper
    i = 20              # projection_horizon -> chosen arbitrarily
    n1 = 8              # relevant_subspace_dim -> chosen
    models = []

    # Loop accross epochs
    for ei, (Ye, Ze) in enumerate(zip(Y[:10], Z[:10])):     # just use first 10 for debugging       # noqa
        tstart = time.time()
        print(f"Processing epoch nbr: {ei}")
        Xh_i, Xh_ip1, Yi, Zi = estimate_latent_state(Ye, Ze, i=i, n1=n1)
        A_11, C_y, C_z = compute_parameters(Xh_i, Xh_ip1, Yi, Zi)
        Wi, Vi = compute_residuals(Xh_i, Xh_ip1, A_11, C_y, Yi)
        ncov = noise_cov_statistics(Wi, Vi)
        P, K = compute_kalman_gain(A_11, C_y, ncov)
        models.append(
            {
                'Xh_i': Xh_i,
                'Xh_ip1': Xh_ip1,
                'Yi': Yi,
                'Zi': Zi,
                'A_11': A_11,
                'C_y': C_y,
                'C_z': C_z,
                'Wi': Wi,
                'Vi': Vi,
                'P': P,
                'K': K
            }
        )

        print(f"Finished after {time.time() - tstart} seconds")

    return models


def estimate_latent_state(Ye, Ze, i=20, n1=10):
    """ Estimate the latent state given data of a single epoch

    Parameters
    ----------
    Ye : np.array (n_channels x n_times)
        neural (eeg) activity
    Ze : np.array (n_channels x n_times)
        behavioral activity
    i : int
        projection horizon for defining how to project past neural activity
        on future behavioral activity
    n1 : int
        number of channels of the hidden neural state

    Returns
    -------
    Xh_i : np.array (n_channels x (n_times - 2 * i + 1))
        estimations of the current hidden state for all n_times - 2 * i + 1
        support points. I.e. all time points in data which allow for a horizon
        of i to the past and the future
    Xh_ip1 : np.array (n_channels x (n_times - 2 * i + 1))
        estimations of the next hidden state for all n_times - 2 * i + 1
        support points. I.e. all time points in data which allow for a horizon
        of i to the past and the future
    Yi : np.array ()
        current neural state for all n_times - 2 * i + 1
        support points. I.e. all time points in data which allow for a horizon
        of i to the past and the future
    Zi : np.array ()
        current behavioral state for all n_times - 2 * i + 1
        support points. I.e. all time points in data which allow for a horizon
        of i to the past and the future
    """
    ny = Ye.shape[0]
    nz = Ze.shape[0]

    # extend matrices to supervectors
    Yv = np.hstack(tuple(Ye.T))
    Zv = np.hstack(tuple(Ze.T))

    Yp = np.vstack([Yv[n * ny:(i - 1 + n) * ny]
                    for n in range(Ye.shape[1] - 2 * i + 2)]).T

    Yp_plus = np.vstack([Yv[n * ny:(i + n) * ny]
                         for n in range(Ye.shape[1] - 2 * i + 2)]).T

    Zf = np.vstack([Zv[(i + n) * nz:(2 * i - 1 + n) * nz]
                    for n in range(Ze.shape[1] - 2 * i + 2)]).T

    Zf_minus = np.vstack([Zv[(n + i + 1) * nz:(2 * i - 1 + n) * nz]
                         for n in range(Ze.shape[1] - 2 * i + 2)]).T

    Yi = Yp[-ny:, :]
    Zi = Zf[:nz, :]

    Zh_f = Zf.dot(Yp.T.dot((1 / (Yp.dot(Yp.T))).dot(Yp)))
    Zh_f_minus = Zf_minus.dot(
        Yp_plus.T.dot(
            (1 / (Yp_plus.dot(Yp_plus.T))).dot(Yp_plus)))

    # SVD
    U, S, V = np.linalg.svd(Zh_f)

    # Compute observability matrix and latent subspace for relevant components
    # only (relevant for the observed behavior) -> cut off based on n1
    G_zi = U[:, :n1].dot(np.diag(np.sqrt(S[:n1])))
    Xh_i = np.linalg.pinv(G_zi).dot(Zh_f)

    # Estimate next hidden state value
    G_zim1 = G_zi[nz:(i - 1) * nz]
    Xh_ip1 = np.linalg.pinv(G_zim1).dot(Zh_f_minus)

    return (Xh_i, Xh_ip1, Yi, Zi)


def compute_parameters(Xh_i, Xh_ip1, Yi, Zi):

    A_11 = Xh_ip1.dot(np.linalg.pinv(Xh_i))
    C_z = Zi.dot(np.linalg.pinv(Xh_i))
    C_y = Yi.dot(np.linalg.pinv(Xh_i))

    return (A_11, C_y, C_z)


def compute_residuals(Xh_i, Xh_ip1, A, C_y, Yi):
    Wi = Xh_ip1 - A.dot(Xh_i)
    Vi = Yi - C_y.dot(Xh_i)

    return (Wi, Vi)


def noise_cov_statistics(Wi, Vi):
    """
    NOTE: j == number of time points used for estimation
    -> nt - 2 * projection_horizon +1

    """
    ncov = 1 / Wi.shape[0] * np.vstack([Wi, Vi]).dot(np.vstack([Wi, Vi]).T)

    return ncov


def compute_kalman_gain(A, C_y, ncov):

    # Start with steady-state solution of the Ricatti equation
    # Paper eq 47 in Supplementary note 5
    x_dim = A.shape[0]
    Q = ncov[:x_dim, :x_dim]
    R = ncov[x_dim:, x_dim:]
    S = ncov[:x_dim, x_dim:]
    E = np.eye(x_dim)

    # eigenvalues of A should be outside the unit disc
    try:
        P_k_km1 = scipy.linalg.solve_discrete_are(A, C_y.T, Q, R, e=E, s=S)
    except Exception as e:
        print(e)
        pdb.set_trace()

    # Kalman gain
    K = (A.dot(P_k_km1.dot(C_y.T)) + S).dot(
        np.linalg.pinv(C_y.dot(P_k_km1.dot(C_y.T)) + R)
    )

    return (P_k_km1, K)


if __name__ == '__main__':
    epo = mne.read_epochs('psid_sample_epo.fif')

    data = split_to_psid_model(epo)

    data = normalize_and_fill_nan(data)

    models = compute_psid_models(data)
