""" Different metrics.
"""
import numpy as np


def calc_acc(y_pred, y_gt):
    """Compute spatial accuracy in terms of the error
        between predicted and ground-truth eye gaze.

    Args:
        y_pred: A nx2 numpy array of predicted eye gaze.
        y_gt: A nx2 numpy array of ground-truth eye gaze.

    Returns:
        A float with spatial accuracy.
    """
    err = y_pred - y_gt
    return np.mean(np.hypot(err[:, 0], err[:, 1]))
