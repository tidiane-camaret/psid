import mne

data = mne.read_epochs('data/VP6_epo.fif')
data.crop(0, 6)
