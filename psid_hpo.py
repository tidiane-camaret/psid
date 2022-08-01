from bohb import BOHB
import bohb.configspace as cs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from psid_accuracy import data_reconstruction,latent_to_features, psid_metrics
import contextlib
from contextlib import contextmanager
import numpy as np
import mne
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, make_scorer
import math
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def psid_eval(
            n1_ratio,
            nx,
            i_psid_ratio,
             ):

    
    print("nx : ", nx)
    n1 = int(math.ceil(nx*n1_ratio))
    print("n1 : ", n1)

    exp = "1"

    freq_ranges = [(5, 8), (8, 13), (13, 30)]#, (128, 132)]
    freq_nbs = [1, 3, 3]#, 1]

    fb = []
    for i_, f in enumerate(freq_ranges):
        points = np.linspace(f[0], f[1], num=freq_nbs[i_] + 1)
        fb += [(points[i], points[i + 1]) for i in range(len(points) - 1)]

    epochs = mne.read_epochs("data/VP" + exp + "_epo.fif")
    sfreq = epochs.info["sfreq"]
    ica_model = mne.preprocessing.read_ica("data/VP" + exp + "_ica.fif")
    
    i_psid = math.floor(sfreq*i_psid_ratio)
    print("i_psid : ", i_psid)

    with suppress_stdout():
        result_dicts = data_reconstruction(epochs, 
                                        ica_model, 
                                        fb,
                                        behav_var="all",
                                        nx=nx,
                                        n1=n1,
                                        i_psid=i_psid
                                        )
    corr, dist, scores, scores_test_only = psid_metrics(result_dicts)

    print("nx : ", nx)
    print("n1 : ", n1)
    print("i_psid : ", i_psid)
    print(np.mean(scores), " ", np.std(scores))

    return 1 - np.mean(scores)


def evaluate(params, n_iterations):
    return psid_eval(**params)

if __name__ == '__main__':

    
    n1_ratio_param = cs.UniformHyperparameter('n1_ratio', 0, 1)
    nx_param = cs.IntegerUniformHyperparameter('nx', 1, 30)
    i_psid_ratio_param = cs.UniformHyperparameter('i_psid_ratio', 0.01, 0.05, log=True)
    
    configspace = cs.ConfigurationSpace([n1_ratio_param, nx_param, i_psid_ratio_param])
    
    """
    alpha = cs.UniformHyperparameter('alpha', 0.01, 0.1, log=True)
    beta = cs.UniformHyperparameter('beta', 0.01, 0.1, log=True)
    configspace = cs.ConfigurationSpace([alpha, beta])
    """
    
    opt = BOHB(configspace, evaluate, max_budget=10, min_budget=1, n_proc=1)

    logs = opt.optimize()
    print(logs)
    """
    psid_eval(n1=5,
            i_psid_ratio=0.01,
             )
    """