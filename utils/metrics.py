import numpy as np

def calc_acc(y_pred, y_gt):
    err = y_pred - y_gt
    return np.mean(np.hypot(err[:, 0], err[:, 1]))

def calc_prec(y_pred, y_gt):
    err = y_pred - y_gt
    return np.mean(np.abs(err))