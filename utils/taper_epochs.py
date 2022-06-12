# Some code to taper end of data

import numpy as np


def taper_epoch_bounds(epo_data: np.ndarray,
                       taper_half_width: int = 200
                       ) -> np.ndarray:
    """ For down the beginning and end of each appoch to 0, --> this decreases
    effects of boundry artefacts when operating on concatenated epochs

    Parameters
    ----------
    epo_data : np.ndarray
        data from an mne epochs object, (n_epos, n_channels, n_times)
    taper_half_width : int
        number of half the time steps to consider for the taper -> this
        is the amount of steps at the end and at the beginning which will
        be tapered

    Returns
    -------
    tapered_data : np.ndarray
        same as epo_data but beginning and end tapered with a hanning window
    """

    taper = np.hanning(taper_half_width * 2)
    taper_start = taper[:taper_half_width]
    taper_end = taper[taper_half_width:]

    stretched_taper = np.hstack(
        [taper_start, np.ones(epo_data.shape[-1] - len(taper)), taper_end]
    )

    tapered_data = np.einsum('ijk,k->ijk', epo_data, stretched_taper)

    return tapered_data


def concat_epochs_data(epo_data: np.ndarray, taper_down: bool = True):
    if taper_down:
        return np.hstack(taper_epoch_bounds(epo_data))
    else:
        return np.hstack(epo_data)


if __name__ == '__main__':
    import mne
    epochs = mne.read_epochs("data/VP1_epo.fif")
    epo_data = epochs.get_data()