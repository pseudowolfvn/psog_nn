import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler

from ml.utils import filter_outliers
from utils.utils import find_filename, find_record_dir

# TODO: move the function to ml.load_data module
# and adapt related functions to use it
def get_subj_data(root, subj_id, data_id, include_outliers=True):
    subj_dir = find_record_dir(root, subj_id)
    subj_root = os.path.join(root, subj_dir)

    data_name = find_filename(
        subj_root, '',
        beg='', end=data_id + '.csv'
    )
    print('Loading data for subject:', subj_root)
    print('Data filename:', data_name)
    data_path = os.path.join(subj_root, data_name)
    data = pd.read_csv(data_path, sep='\t')

    if not include_outliers:
        # the incomplete outlier detection is used for
        # backward compatibility with old results of IVT algorithm
        data = data[data.val != 4]

        # TODO: check that complete outliers detection gives
        # reasonable results by visual inspection of the difference
        # it makes and switch to using it
        # matlab_name = find_filename(subj_root, '', beg='Recording_', end='Velocity.mat')
        # matlab_path = os.path.join(subj_root, matlab_name)
        # matlab_data = loadmat(matlab_path)

        # data = filter_outliers(data, matlab_data)

    return data

def get_model_path():
    return os.path.join(
        'em_classification', 'psog', '1dcnn_model_eps_0.4.pt'
    )

def convert_to_window(X, y, w, future_samples=True):
    X_w = np.zeros((X.shape[0] - 2*w, 2*w + 1, X.shape[1]), dtype=np.float32)

    if future_samples:
        for i in range(w, X.shape[0] - w):
            X_w[i - w, :, :] = X[i - w: i + w + 1]
        y = y[w: y.shape[0] - w]
    else:
        for i in range(2*w, X.shape[0]):
            X_w[i - 2*w, :, :] = X[i - 2*w : i + 1]
        y = y[2*w: y.shape[0]]

    return X_w, y

def to_channel_first(X):
    def convert(X):
        N, L, C = X.shape
        return np.reshape(X, (N, C, L))
    
    if type(X) is not list:
        X = [X]

    return [convert(x) for x in X]

def scale_data(X):
    normalizer = RobustScaler()
    normalizer.fit(X)

    X = normalizer.transform(X)
    return X