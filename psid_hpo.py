import math
import sys
import os
import argparse
import numpy as np

from contextlib import contextmanager

import mne

from bohb import BOHB
import bohb.configspace as cs

from psid_accuracy import signal_to_features, psid_metrics

@contextmanager 
### avoids console output during psd_array_multitaper
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

    freq_ranges = [(5, 8), (8, 13), (13, 30)]
    freq_nbs = [1, 3, 3]

    fb = []
    for i_, f in enumerate(freq_ranges):
        points = np.linspace(f[0], f[1], num=freq_nbs[i_] + 1)
        fb += [(points[i], points[i + 1]) for i in range(len(points) - 1)]

        epochs = mne.read_epochs(os.path.join(data_dir,"VP" + str(exp_idx) + "_epo.fif"))
        sfreq = epochs.info["sfreq"]
        ica_model = mne.preprocessing.read_ica(os.path.join(data_dir,"VP" + str(exp_idx) + "_ica.fif"))

    
    i_psid = math.floor(sfreq*i_psid_ratio)
    print("i_psid : ", i_psid)

    with suppress_stdout():
        result_dicts = signal_to_features(epochs, 
                                        ica_model, 
                                        fb,
                                        behav_var="all",
                                        nx=nx,
                                        n1=n1,
                                        i_psid=i_psid
                                        )
    corr, dist, lda_scores, xgb_scores, xgb_fi, lda_scores_test_only = psid_metrics(result_dicts, nx)

    print("nx : ", nx)
    print("n1 : ", n1)
    print("i_psid : ", i_psid)
    print(np.mean(lda_scores), " ", np.std(lda_scores))

    return 1 - np.mean(lda_scores)


def evaluate(params, n_iterations):
    return psid_eval(**params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PSID_hpo')

    parser.add_argument('--exp_idx', type=int, default=1, help='idex of the experiment to run')
    parser.add_argument('--data_dir', type=str, default='data')

    args = parser.parse_args()

    data_dir = args.data_dir
    exp_idx = args.exp_idx
    
    n1_ratio_param = cs.UniformHyperparameter('n1_ratio', 0, 1)
    nx_param = cs.IntegerUniformHyperparameter('nx', 1, 30)
    i_psid_ratio_param = cs.UniformHyperparameter('i_psid_ratio', 0.01, 0.05, log=True)
    
    
    configspace = cs.ConfigurationSpace([n1_ratio_param, nx_param, i_psid_ratio_param])
    
    
    opt = BOHB(configspace, evaluate, max_budget=1, min_budget=1, n_proc=1)

    logs = opt.optimize()
    print(logs)
