import sys
import matplotlib.pyplot as plt
import pickle
import mne
import xgboost as xgb
from sklearn.metrics import accuracy_score
sys.path.append("..")
from utils.csp_example import custom_CSP, LDA, Pipeline
from utils.features_utils import CSP_LDA_features, block_crossval

csp_decoder = custom_CSP(n_components=4, log=True,
                         cov_kwargs={'method': 'ledoit_wolf'})

model = Pipeline(steps=[('spatial_filtering', csp_decoder),
                        ('decoder', LDA())])

#exps = [str(n) for n in range(3, 11)]
exps = ["8"]

for exp in exps:
    epochs = mne.read_epochs("data/VP" + exp + "_300hz_epo.fif").crop(0, 6)
    ica_model = mne.preprocessing.read_ica("data/VP" + exp + "_ica.fif")

    bscores_lng = []
    bscores_lno = []

    search_range = range(128,132,1) #range(5, 35, 2)

    for tfa in search_range:
        tfb = tfa + 2
        print("PROCESSING RANGE " + str(tfa) + ", " + str(tfb))
        """
        X, y, blocks_idx = CSP_LDA_features(epochs, ica_model, tf=[tfa, tfb])

        with open('results/csp_' + exp + '.pickle', 'wb') as handle:
            pickle.dump((X, y, blocks_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)

        """
        with open('results/csp_'+exp+'.pickle', 'rb') as handle:
            X, y, blocks_idx = pickle.load(handle)
        

        bscores_lng.append(block_crossval(X, y, model, blocks_idx, metric=accuracy_score, leaveP=True))
        bscores_lno.append(block_crossval(X, y, model, blocks_idx, metric=accuracy_score, leaveP=False))

    with open('results/csp_' + exp + '_lng_acc.pickle', 'wb') as handle:
        pickle.dump(bscores_lng, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('results/csp_' + exp + '_lno_acc.pickle', 'wb') as handle:
        pickle.dump(bscores_lno, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig, axs = plt.subplots(1,2)
axs[0].boxplot(bscores_lng,[str(n)+ " " + str(n+2) for n in search_range])
axs[0].set_title("csp accuracy for different freq ranges using leave_p_groups crossval")
axs[1].boxplot(bscores_lno,[str(n)+ " " + str(n+2) for n in search_range])
axs[1].set_title("csp accuracy for different freq ranges using my_leave_n_out crossval")
plt.show()
