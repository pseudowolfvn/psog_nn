import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preproc.psog import PSOG
from .utils import filter_outliers, normalize


def get_subj_data(subj_root, sensor=None):
    if sensor is None:
        sensor = PSOG()

    data_name = Path(subj_root).name + '_' + sensor.arch + '.csv'
    data_path = os.path.join(subj_root, data_name)
    data = pd.read_csv(data_path, sep='\t')

    data = filter_outliers(data)

    X = data[sensor.get_names()].values
    y = data[['pos_x', 'pos_y']].values

    return X, y

def get_data(root, subj_ids=None):
    X_data = []
    y_data = []
    sensor = PSOG()
    for dirname in os.listdir(root):
        if subj_ids is not None and dirname not in subj_ids:
            continue
        subj_root = os.path.join(root, dirname)
        X, y = get_subj_data(subj_root, sensor)

        X_data.extend(X)
        y_data.extend(y)

    return np.array(X_data), np.array(y_data)

def reshape_into_grid(X_train, X_val, X_test):
    X_train = X_train.reshape((X_train.shape[0], 3, 5, 1))
    X_val = X_val.reshape((X_val.shape[0], 3, 5, 1))
    X_test = X_test.reshape((X_test.shape[0], 3, 5, 1))
    return X_train, X_val, X_test

def get_general_data(root, train_subjs, test_subjs, arch):
    X_train, y_train = get_data(root, train_subjs)
    X_test, y_test = get_data(root, test_subjs)
    X_train, X_test = normalize(X_train, X_test, train_subjs, arch)

    # train_val_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_specific_data(root, test_subjs, subj, arch, load_normalizer):
    X_train, y_train = get_data(root, subj)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.7, random_state=42)
    X_train, X_test = normalize(X_train, X_test, test_subjs, arch, load_normalizer)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def split_test_from_all(test_subjs):
    data = [str(i) for i in range(1, 23 + 1) if i != 9]
    train_subjs = []
    for subj in data:
        if subj not in test_subjs:
            train_subjs.append(subj)
    return train_subjs, test_subjs