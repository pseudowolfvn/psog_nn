""" Read dataset in ready-to train/test format.
"""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.utils import filter_outliers, normalize
from preproc.psog import PSOG
from utils.utils import list_if_not, deg_to_pix


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
    # when only one id is provided we shouldn't
    # consider it as iterable but enclose in the list
    if subj_ids is not None:
        subj_ids = list_if_not(subj_ids)

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
    X_train, X_test = normalize(X_train, X_test, arch, train_subjs)

    # train_val_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_specific_data(root, subj, arch, train_subjs=None):
    X_train, y_train = get_data(root, subj)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.7, random_state=42)
    X_train, X_test = normalize(X_train, X_test, arch, train_subjs)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_stimuli_pos(root, subj):
    subj_root = os.path.join(root, subj)
    subj = Path(subj_root).name

    data_path = 'Stimulus.xml'

    tree = ET.parse(os.path.join(subj_root, data_path))
    root = tree.getroot()

    stimuli_pos = []
    for position in root.iter('Position'):
        x = int(position.find('X').text)
        y = int(position.find('Y').text)
        stimuli_pos.append((x, y))

    # stimuli_pos = np.array(stimuli_pos)
    return stimuli_pos

def get_calib_like_data(root, subj, arch):
    subj_root = os.path.join(root, subj)
    stimuli_pos = get_stimuli_pos(root, subj)
    calib_pos = sorted(list(set(stimuli_pos)))
    calib_pos = [
        calib_pos[0],
        calib_pos[2],
        calib_pos[4],
        calib_pos[8],
        calib_pos[10],
        calib_pos[12],
        calib_pos[16],
        calib_pos[18],
        calib_pos[20]
    ]

    X_train, y_train = get_subj_data(subj_root)

    train_ind = []
    test_ind = []
    for ind, pos in enumerate(y_train):
        posx, posy = deg_to_pix(pos)
        calib_point = False
        for calib in calib_pos:
            x, y = calib
            dist = np.hypot(posx - x, posy - y)
            if dist < 35.:
                calib_point = True
                break
        if calib_point:
            train_ind.extend([ind])
        else:
            test_ind.extend([ind])

    X_test = X_train[test_ind]
    y_test = y_train[test_ind]

    X_train = X_train[train_ind]
    y_train = y_train[train_ind]

    X_train, X_test = normalize(X_train, X_test, arch)

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42)

    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_test_from_all(test_subjs):
    data = [str(i) for i in range(1, 23 + 1)]
    train_subjs = []
    for subj in data:
        if subj not in test_subjs:
            train_subjs.append(subj)
    return train_subjs, test_subjs
