import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from utils.utils import find_filename, find_record_dir


def get_ivt_gaze_subj_data(root, subj_id):
    subj_dir = find_record_dir(root, subj_id)
    subj_root = os.path.join(root, subj_dir)

    data_name = find_filename(subj_root, 'Signal.tsv', beg='DOT', end='.tsv')

    data = pd.read_csv(os.path.join(subj_root, data_name), sep='\t')

    mask = ['Timestamp', 'GazePointXLeft', 'GazePointYLeft']
    return data[data.ValidityLeft != 4][mask]

def get_subj_data(root, subj_id, data_id):
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