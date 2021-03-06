""" Read dataset in ready-to train/test format.
"""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from ml.utils import filter_outliers, normalize
from preproc.psog import PSOG
from utils.utils import list_if_not, deg_to_pix, find_filename, extract_subj_id_from_dir


def get_subj_data(subj_root, sensor=None, with_shifts=False, with_time=False, data_suffix='randn_v2'):
    # TODO: update comment for new parameters
    """Get data for provided subject.

    Args:
        subj_root: A string with full path to directory
            with subject's data stored in .csv file. 

    Return:
        A tuple of array with PSOG sensor raw outputs,
            array with corresponding ground-truth eye gazes.
    """
    if sensor is None:
        sensor = PSOG()

    data_name = find_filename(subj_root, '', beg='Record', end='_' + data_suffix + '.csv')
    data_path = os.path.join(subj_root, data_name)
    data = pd.read_csv(data_path, sep='\t')

    matlab_name = find_filename(subj_root, '', beg='Recording_', end='Velocity.mat')
    matlab_path = os.path.join(subj_root, matlab_name)
    matlab_data = loadmat(matlab_path)

    data = filter_outliers(data, matlab_data)

    X_cols = (['time'] if with_time else []) + \
        sensor.get_names() + \
        (['sh_hor', 'sh_ver'] if with_shifts else [])

    X = data[X_cols].values
    y = data[['pos_x', 'pos_y']].values

    return X, y

def get_data(root, subj_ids=None, with_shifts=False, with_time=False, data_suffix='randn_v2'):
    # TODO: update comment for new parameters
    """Get data for the provided list of subjects.

    Args:
        root: A string with path to dataset.
        subj_ids: A list with subjects ids to get data for if provided,
            otherwise get data for the whole dataset.

    Return:
        A tuple of array with PSOG sensor raw outputs,
            array with corresponding ground-truth eye gazes.
    """
    # when only one id is provided we shouldn't
    # consider it as iterable but enclose in the list
    if subj_ids is not None:
        subj_ids = list_if_not(subj_ids)

    X_data = []
    y_data = []
    sensor = PSOG()
    for dirname in os.listdir(root):
        if subj_ids is not None and \
                extract_subj_id_from_dir(dirname) not in subj_ids:
            continue
        subj_root = os.path.join(root, dirname)
        X, y = get_subj_data(
            subj_root, sensor,
            with_shifts=with_shifts, with_time=with_time,
            data_suffix=data_suffix
        )

        X_data.extend(X)
        y_data.extend(y)

    return np.array(X_data), np.array(y_data)

def reshape_into_grid(X_train, X_val, X_test):
    """Reshape 1x15 flatten data into 3x5 grid.

    Args:
        X_train: An array of training set values with
            PSOG sensor raw outputs.
        X_val: An array with validation set.
        X_test: An array with test set.

    Returns:
        A tuple with reshaped training, validation and test sets.
    """
    X_train = X_train.reshape((X_train.shape[0], 1, 3, 5))
    X_val = X_val.reshape((X_val.shape[0], 1, 3, 5))
    X_test = X_test.reshape((X_test.shape[0], 1, 3, 5))
    return X_train, X_val, X_test

# TODO: choose a better name
def get_general_data(root, train_subjs, test_subjs, arch):
    """Get data when training and test sets consist of different subjects.

    Args:
        root: A string with path to dataset.
        train_subjs: A list of subjects ids to train on.
        test_subjs: A list of subjects ids to test on.
        arch: A string with model architecture id.

    Returns:
        A tuple of 6 arrays: 
            array of training set values with PSOG sensor raw outputs, array of
            training set values with corresponding ground-truth eye gazes,
            two arrays of corresponding validation set values,
            two arrays of corresponding test set values.
    """
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
    """Get data when training and test sets are
        split from the same provided subject.

    Args:
        root: A string with path to dataset.
        subj: A string with specific subject id.
        arch: A string with model architecture id.
        train_subjs: A list of subjects ids model
            was pre-trained on if provided.

    Returns:
        The same as in ml.load_data.get_general_data().
    """
    X_train, y_train = get_data(root, subj)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.7, random_state=42)
    X_train, X_test = normalize(X_train, X_test, arch, train_subjs)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_shifts_split_data(root, subj, arch, train_subjs=None):
    X_train, y_train = get_data(root, subj, with_shifts=True)

    mask = np.rint(X_train[:, -2:] / 0.5)
    train_ind = np.where((mask % 2).any(axis=1))[0]
    test_ind = []
    for i in range(X_train.shape[0]):
        if i not in train_ind:
            test_ind.append(i)

    X_test = X_train[:, :-2][test_ind]
    y_test = y_train[test_ind]

    X_train = X_train[:, :-2][train_ind]
    y_train = y_train[train_ind]

    X_train, X_test = normalize(X_train, X_test, arch, train_subjs)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    print(X_train.shape, X_val.shape, X_test.shape)
    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_shifts_outer_split_data(root, subj, arch,
        test_rad_gt, test_rad_lt, train_rad=1.0, train_subjs=None):
    dist = lambda x, y: np.sqrt(x**2 + y**2)

    X_data, y_data = get_data(root, subj, with_shifts=True)

    train_ind = np.where([
        dist(x, y) <= train_rad 
        for x, y in X_data[:, -2:]]
    )[0]

    X_train = X_data[:, :-2][train_ind]
    y_train = y_data[train_ind]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42)

    if test_rad_lt == train_rad:
        X_test = X_temp
        y_test = y_temp
    else:
        test_ind = np.where([
            dist(x, y) > test_rad_gt and dist(x, y) <= test_rad_lt
                for x, y in X_data[:, -2:]
        ])[0]
        X_test = X_data[:, :-2][test_ind]
        y_test = y_data[test_ind]

    X_train, X_test = normalize(X_train, X_test, arch, train_subjs)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    print(X_train.shape, X_val.shape, X_test.shape)
    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def make_get_shifts_outer_split_data(test_rad_gt, test_rad_lt, train_rad=1.0):
    def f(root, subj, arch, train_subjs=None):
        return get_shifts_outer_split_data(
            root, subj, arch,
            test_rad_gt=test_rad_gt, test_rad_lt=test_rad_lt,
            train_rad=train_rad, train_subjs=train_subjs
        )
    return f

def get_ratio_data(root, subj, arch, train_subjs=None, ratio=1.0, seed=42):
    X_train, y_train = get_data(root, subj)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=seed)
    X_train, X_test = normalize(X_train, X_test, arch, train_subjs)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.143, random_state=seed)

    train_size = int(round(ratio * X_train.shape[0]))
    print('DEBUG: ', train_size, 'out of', X_train.shape[0])
    X_train = X_train[:train_size, :]
    y_train = y_train[:train_size, :]

    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def make_get_ratio_data(ratio, seed=42):
    def f(root, subj, arch, train_subjs=None):
        return get_ratio_data(root, subj, arch,
            train_subjs=train_subjs, ratio=ratio, seed=seed)
    return f

def default_source_if_none(data_source):
    """Get default data source function object.

    Args:
        data_source: A function object that is used to get data.

    Returns:
        A function object passed by 'data_source' arg if it's not None
            otherwise default data source function.
    """
    if data_source is None:
        data_source = get_specific_data
    return data_source

def get_stimuli_pos(root, subj):
    """Get stimuli positions in pixels used for provided subject's recording.

    Args:
        root: A string with path to dataset.
        subj: A string with specific subject id.

    Returns:
        An array of tuples, each of one represents point on the screen.
    """
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

def get_calib_like_data(root, subj, arch, train_subjs=None):
    """Get data with calibration-like training set
        distribution for provided subject.

    Args:
        root: A string with path to dataset.
        subj: A string with specific subject id.
        arch: A string with model architecture id.
        train_subjs: None, ignored.

    Returns:
        The same as in ml.load_data.get_specific_data().
    """
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
    """Split the whole dataset into train and
        test sets of subjects ids with later one provided.

    Args:
        test_subjs: A list with test set subjects ids.

    Returns:
        A tuple of array with training set subjects ids,
            array with test set subjects ids.
    """
    data = [str(i) for i in range(1, 23 + 1)]
    train_subjs = []
    for subj in data:
        if subj not in test_subjs:
            train_subjs.append(subj)
    return train_subjs, test_subjs

def default_split_if_none(split_source):
    if split_source is None:
        split_source = get_default_subjs_split
    return split_source

def get_default_subjs_split():
    """Get default split into test set subjects ids chunks.

    Returns:
        A list of lists, each of one represents test set subjects ids chunk.
    """
    return [
        ['1', '2', '3', '4'],
        ['5', '6', '7', '8'],
        ['9', '10', '11', '12'],
        ['13', '14', '15', '16'],
        ['17', '18', '19', '20'],
        ['21', '22', '23']
    ]

def get_loo_subjs_split():
    return [[str(i)] for i in range(1, 24)]

def find_train_test_split(subj):
    """Get split of the whole dataset into train and
        test sets of subjects ids with only one subject id provided.

    Args:
        subj: A string with specific subject id.

    Returns:
        The same as in ml.load_data.split_test_from_all().
    """
    subjs_split = get_default_subjs_split()

    for split in subjs_split:
        if subj in split:
            return split_test_from_all(split)
    return None

def get_loo_subjs_split():
    return [[str(i)] for i in range(1, 41)]