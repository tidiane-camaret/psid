import sys
import matplotlib.pyplot as plt
import pickle
import mne
import xgboost as xgb

sys.path.append("..")
from utils.csp_example import custom_CSP, LDA, Pipeline
from utils.features_utils import CSP_LDA_EVAL, random_crossval, block_crossval

model = xgb.XGBClassifier(max_depth=5,
                          n_estimators=10,
                          n_jobs=3,
                          eval_metric="logloss",
                          use_label_encoder=False)

csp_decoder = custom_CSP(n_components=4, log=True,
                         cov_kwargs={'method': 'ledoit_wolf'})

model = Pipeline(steps=[('spatial_filtering', csp_decoder),
                        ('decoder', LDA())])

exps = [str(n) for n in range(1, 3)]

for exp in exps:
    epochs = mne.read_epochs("data/VP" + exp + "_epo.fif").crop(0, 6)
    ica_model = mne.preprocessing.read_ica("data/VP" + exp + "_ica.fif")

    rscores = []
    bscores = []

    search_range = range(5, 35, 2)

    for tfa in search_range:
        tfb = tfa + 2
        print("PROCESSING RANGE " + str(tfa) + ", " + str(tfb))

        X, y, blocks_idx = CSP_LDA_EVAL(epochs, ica_model, tf=[tfa, tfb])

        with open('results/csp_' + exp + '.pickle', 'wb') as handle:
            pickle.dump((X, y, blocks_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)

        """
        with open('results/csp_'+exp+'.pickle', 'rb') as handle:
            X, y, blocks_idx = pickle.load(handle)
        """

        bscores.append(block_crossval(X, y, model, blocks_idx))

    with open('results/csp_' + exp + '_acc.pickle', 'wb') as handle:
        pickle.dump(bscores, handle, protocol=pickle.HIGHEST_PROTOCOL)


plt.boxplot(bscores)
plt.show()
