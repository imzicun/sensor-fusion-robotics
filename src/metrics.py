import numpy as np

def rmse(true, est):
    true = np.asarray(true)
    est = np.asarray(est)
    return float(np.sqrt(np.mean((true - est) ** 2)))
