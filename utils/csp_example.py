# Here is some example code pices combined for a standalone use

import pdb
import mne
import json
import numpy as np

from scipy import linalg
from mne.cov import _regularized_covariance

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score, make_scorer, accuracy_score
from sklearn.model_selection import (StratifiedKFold,
                                     permutation_test_score)

from scipy import signal

import warnings


# ----- Custom aux stuff
class custom_CSP(mne.decoding.CSP):
    """
    Note: An individual overwrite for the CSP class is necessary if mini epochs
    (see below) are used
    """
    def __init__(self, cov_kwargs={}, **kwargs):
        self.cov_kwargs = cov_kwargs
        super().__init__(**kwargs)

    def compute_covariances(self, X, y):
        assert len(y.shape) == 1

        ix_c1 = np.where(y == np.unique(y)[0])[0]
        ix_c2 = np.where(y == np.unique(y)[1])[0]

        if len(ix_c1) + len(ix_c2) != len(y):
            raise Exception('Make sure there are only 2 classes and they'
                            ' are encoded in y with 0 and 1')

        # assume that X is an array of sub epochs with various lengths
        Cxx1 = mne.compute_covariance(list(X[ix_c1]), **self.cov_kwargs)
        Cxx2 = mne.compute_covariance(list(X[ix_c2]), **self.cov_kwargs)
        Cavg = mne.compute_covariance(list(X), **self.cov_kwargs)

        return Cxx1, Cxx2, Cavg

    def fit(self, X, y):
        Cxx1, Cxx2, Cavg = self.compute_covariances(X, y)
        evals, evecs = linalg.eig(Cxx2.data - Cxx1.data, Cavg.data)
        evals = evals.real
        evecs = evecs.real

        # sort vectors
        ix = np.argsort(np.abs(evals))[::-1]
        # sort eigenvectors
        evecs = evecs[:, ix].T

        # spatial patterns
        self.patterns_ = linalg.pinv(evecs).T  # n_channels x n_channels
        self.filters_ = evecs  # n_channels x n_channels

        return self

    def transform(self, X):
        """
        X would be the array or list of mini epochs here. Note that in the
        original implementation of Sebastian's, he worked on precomputed
        covariance matrices to speed up the process

        --> This is not done here for the sake of clarity/readability!

        """
        pick_filters = self.filters_[:self.n_components]

        # TODO: Fixme --> this part of the projection still fails
        # The problem is, that still each filtered epoch is of different
        # length in terms of time ... -> try to understand what the output
        # of Sebastian's CSP would yield if it was functional...
        #
        # --> this adds about 8s per transform call, but makes it much more
        # readable what is actually going on, can be optimized most likely
        Cx = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            # a cov matrix in the csp filter space
            for me in X:
                cx = np.dot(
                    np.dot(
                        pick_filters, mne.compute_covariance(
                            me,
                            **self.cov_kwargs,
                            verbose=False).data
                    ),
                    pick_filters.T
                )
                Cx.append(np.diag(cx))

        Cx = np.stack(Cx, axis=0)
        if (Cx <= 0).any():
            pdb.set_trace()


        log = True if self.log is None else self.log
        if log:
            Cx = np.log(Cx)

        if len(Cx.shape) == 1:
            Cx = Cx.reshape(-1, 1)

        return Cx


def auc_scoring(y, y_pred):
    auc_threshold = np.median(y)
    ix_true = np.where(y > auc_threshold)[0]
    y_binary = np.zeros(y.shape)
    y_binary[ix_true] = 1
    score = roc_auc_score(y_binary, y_pred - auc_threshold)
    return score


def ica_get_unmixing(ica_model):

    exclude = ica_model.exclude
    _n_pca_comp = ica_model._check_n_pca_components(ica_model.n_pca_components)
    n_components = ica_model.n_components_
    sel_keep = np.arange(n_components)
    if exclude not in (None, []):
        sel_keep = np.setdiff1d(np.arange(n_components), exclude)

    _n_pca_comp = ica_model._check_n_pca_components(ica_model.n_pca_components)
    n_components = ica_model.n_components_
    unmixing = np.eye(_n_pca_comp)
    unmixing[:n_components, :n_components] = ica_model.unmixing_matrix_
    unmixing = np.dot(unmixing, ica_model.pca_components_[:_n_pca_comp])

    return unmixing[sel_keep, :]


def project_ica(epochs, ica_model):
    """
    projects epochs to the ica space
    """

    picks = mne.pick_types(epochs.info, meg=False, ref_meg=False,
                           include=epochs.ch_names,
                           exclude='bads')
    data = np.hstack(epochs.get_data()[:, picks])
    # data, _ = self.ica_model._pre_whiten(data, epochs.info, picks)
    data = ica_model._pre_whiten(data)     # new mne version
    data = data.copy()

    if ica_model.pca_mean_ is not None:
        data -= ica_model.pca_mean_[:, None]

    unmixing = ica_get_unmixing(ica_model)

    proj_data = np.dot(unmixing, data)
    proj_data = np.array(
        np.split(proj_data, len(epochs.events), 1))

    # restore as epochs, channels, tsl order
    ch_names = ["ICA_%d" % ix for ix in range(proj_data.shape[1])]
    info_ica = mne.create_info(ch_names, epochs.info['sfreq'], ch_types='eeg')
    epochs_ica = mne.EpochsArray(proj_data, info_ica, tmin=epochs.tmin,
                                 verbose=False)
    return epochs_ica


# --> the original code from Sebastion
def create_subepochs(epochs, subepo_crop_tmin=2,
                     reject=dict(eeg=80e-6),
                     flat=dict(eeg=1e-8),
                     target_frequency=[1, 25],
                     reject_frequency=None,
                     subepoch_length=1,
                     subepoch_tmin=0,
                     subepoch_ival=1,
                     subepoch_overlap=0., verbose=None):
    """
    create subepochs from an mne.Epoch object with this, we avoid throwing
    away entire epochs if only a segment is contaminated.
    Specially meant for copyDraw epochs, which are very long (~7 sec) by design.

    Parameters:
    -----------
    subepo_crop_tmin : int
        time [sec] for t=0 in each epoch, for each the subepochs are generated.
        used for performing filtering later with fearing for border artifacts.
    reject : see mne.Epoch
    flat : see mne.Epoch
    target_frequency: list len=2
        freq band [Hz] in which the output subepochs are filtered
    reject_frequecy: list len=2
        perform artifact rejection in this freq. band
    subepoch_lengh: int
        tmax of each subepoch
    epoch_tmin: int
        tmin of each subepoch
    subepoch_ival: int
        distance between t=0 of each subepoch
    Returns:
    --------
    epochs_array :  list of mne.Epochs
        each element of the array contains the corresponding sub-epoched copyDraw trial

    ev_array :  list of numpy.array
        each element contained the events used to generate the epochs in epochs_array

    """
    # return
    info = epochs.info
    subepo_stop = epochs.times[-1]

    if not target_frequency is None:
        sos_target_band = signal.iirfilter(5, np.array(target_frequency)/(epochs.info['sfreq']/2),
                                           output='sos')
        do_filtering = True
    else:
        do_filtering = False

    if reject_frequency is None:
        reject_frequency = target_frequency
        filter_twice = False
    else:
        filter_twice = True

    if not reject_frequency is None:
        sos_reject = signal.iirfilter(5, np.array(reject_frequency)/(epochs.info['sfreq']/2),
                                      output='sos')

    epochs_array = []
    ev_array = []
    tmin = epochs.tmin
    tmax = epochs.tmax
    # pdb.set_trace()
    for ix_epo, epo in enumerate(epochs.get_data()):
        # get reject trials using "reject band", but dont perform the analysis in
        # this band, in case, the analysis is performed outside that interval

        # Cast the a single epo into a raw array object -> this mask the epoch
        # as if it would be a full trial, hence upon slicing it in "epochs"
        # we get the desired mini/sub epoches
        mini_raw = mne.io.RawArray(epo, info, verbose=False)

        if do_filtering:
            # perform the artifact rejection on the specified frequency band
            # if power is greater than upper threshold or stale
            #
            # --> half updated python breaks here if n_jobs >= 1 do to incompatebility with
            # pickle 5
            mini_rawReject = mini_raw.copy().filter(*reject_frequency, method='iir',
                                                    iir_params={'sos': sos_reject}, n_jobs=1)
        else:
            mini_rawReject = mini_raw.copy()
        mini_rawReject.crop(tmin=-tmin-subepo_crop_tmin)

        # set arbitrary markers to slice the epoch into mini/sub epochs
        ev_dummy = mne.make_fixed_length_events(mini_rawReject, 666, start=subepo_crop_tmin,
                                                stop=subepo_stop-tmin, duration=subepoch_ival,
                                                overlap=subepoch_overlap)

        # the actual mini epoch creation
        mini_epochsRej = mne.Epochs(mini_rawReject, ev_dummy, 666, baseline=None,
                                    reject=reject, flat=flat,
                                    verbose=verbose, tmin=-subepo_crop_tmin, tmax=subepoch_length)

        mini_epochsRej.drop_bad()
        print(f"Dropping in miniEpoch - drop_log: {mini_epochsRej.drop_log}")

        ix_accepted = [i for i, x in enumerate(
            mini_epochsRej.drop_log) if len(x) == 0]

        # pdb.set_trace()
        if len(ix_accepted) > 0:
            if filter_twice:
                ix_accepted = np.unique(ix_accepted)
                ev_dummy = ev_dummy[ix_accepted]
                if do_filtering:
                    mini_raw.filter(*target_frequency, method='iir',
                                    iir_params={'sos': sos_target_band},
                                    n_jobs=1)
                # mini_raw.crop(tmin=-tmin-subepo_crop_tmin)
                mini_epochs = mne.Epochs(mini_raw, ev_dummy, 666,
                                         verbose=False,
                                         reject=None, flat=None,
                                         baseline=None, tmin=subepoch_tmin,
                                         tmax=subepoch_length)
            else:
                mini_epochs = mini_epochsRej

            # if there is at least one good mini epoch, append to the final
            # array, else use the empty list[]
            epochs_array.append(mini_epochs)
            ev_array.append(ev_dummy)
        else:
            epochs_array.append([])
            ev_array.append([])

    return epochs_array, ev_array


def main():
    # ------ data loading
    # Note: Instead of the full 131 epochs, Sebastian sliced each in additional
    # smaller "mini epochs" (1s) which where then considered for a peak-to-peak
    # filter criterion, basically discarding any mini_epoch with a max p2p of
    # 80 uV
    #
    # Note also that these mini_epochs need to be sliced on already filtered
    # epoch data, else the boundry artifacts of the mini_epochs would be to
    # strong
    epochs = mne.read_epochs('sample_epo.fif')

    tf = [25.38, 30.38]         # <-- found by gridsearch

    mini_epochs_list, _ = create_subepochs(epochs, target_frequency=tf,
                                           reject_frequency=[1, 30])
    mini_epochs_list = [e.load_data() for e in mini_epochs_list]

    y = np.asarray(json.load(open('./labels.json', 'r')))
    ica_model = mne.preprocessing.read_ica('./selected_ica.fif')

    # ------ decoding pipeline
    csp_decoder = custom_CSP(n_components=4, log=True,
                             cov_kwargs={'method': 'ledoit_wolf'})
    model = Pipeline(steps=[('spatial_filtering', csp_decoder),
                            ('decoder', LDA())])

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    # ---- Epochs to ICA space
    # In the original script, all fitting and CV was done in ICA space
    # This is replicated here, this includes whitening the data
    ica_mini_epochs = [project_ica(e, ica_model) for e in mini_epochs_list]

    X = np.asarray(ica_mini_epochs, dtype='object')

    splits = list(cv.split(X, y))
    scorer = make_scorer(auc_scoring)
    scores = []

    for isplit, (ix_train, ix_test) in enumerate(splits):

        print("=" * 80)
        print(f"Processing split {isplit + 1}")

        model.fit(X[ix_train], y[ix_train])
        scores.append(scorer(model, X[ix_test], y[ix_test]))

    print(f"Scores: {scores} - {np.mean(scores):.2%}")



if __name__ == '__main__':
    main()
