import sys
import pickle
import mne
import xgboost as xgb

sys.path.append("..")
from utils.features_utils import PSID_features

model = xgb.XGBClassifier(max_depth=5,
                          n_estimators=10,
                          n_jobs=3,
                          eval_metric="logloss",
                          use_label_encoder=False)

bscores = []

exps = [str(n) for n in range(1, 2)]

for exp in exps:
    epochs = mne.read_epochs("data/VP" + exp + "_epo.fif")
    ica_model = mne.preprocessing.read_ica("data/VP" + exp + "_ica.fif")
    X, y, blocks_idx = PSID_features(epochs, ica_model, include_stim=True)

    with open('results/psid_' + exp + '.pickle', 'wb') as handle:
        pickle.dump((X, y, blocks_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)

    X, y, blocks_idx = PSID_features(epochs, ica_model, include_stim=False)

    with open('results/psid_' + exp + '_nostim.pickle', 'wb') as handle:
        pickle.dump((X, y, blocks_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)

    """"
    with open('results/psid_' + exp + '.pickle', 'rb') as handle:
        X, y, blocks_idx = pickle.load(handle)
    print(X.shape, y.shape)

    bscores = block_crossval(X, y, model, blocks_idx, do_feature_imp=True)

    with open('results/psid_' + exp + '_acc.pickle', 'wb') as handle:
        pickle.dump(bscores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """

"""
plt.scatter(x=exps, y=[np.mean(s[0]) for s in bscores])
plt.show()
"""