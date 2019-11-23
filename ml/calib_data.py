import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.load_data import get_subj_data, reshape_into_grid
from ml.utils import robust_scaler
from utils.utils import find_record_dir


def make_calib_grid_specific_data(data_id='randn', calib_grid=29, manual=False):
    def f(root, subj_id, arch, train_subjs):
        return get_specific_data(
            root, subj_id, arch,
            train_subjs=train_subjs,
            data_id=data_id,
            calib_grid=calib_grid,
            manual=manual
        )
    return f

def get_specific_data(root, subj_id, arch, train_subjs=None, data_id='randn', calib_grid=29, manual=False):
    # fix_bounds, stim_pos = get_fix_bounds_stim_pos(root, subj_id)
    print('DEBUG, getting data for:', subj_id, train_subjs, data_id, calib_grid, manual)
    fix_bounds, stim_pos = filter_by_calib_grid(root, subj_id, calib_grid=calib_grid, manual=manual)
    return get_calib_train_data(root, subj_id, arch, fix_bounds, stim_pos, data_id=data_id, train_subjs=train_subjs)

def get_calib_train_data(root, subj_id, arch, fix_bounds, stim_pos, data_id='randn', train_subjs=None):
    subj_dir = find_record_dir(root, subj_id)
    subj_root = os.path.join(root, subj_dir)
    X, y = get_subj_data(subj_root, with_time=True, data_suffix=data_id)
    #####################
    # X, y = normalize_to_neutral(X, y, fix_bounds, stim_pos)
    #####################
    train_inds = []
    test_inds = []
    y_train = np.zeros((0, 2))
    fix_ind = 0
    ind = 0
    # skip data before first calibration target
    while X[ind, 0] < fix_bounds[0][0]:
        ind += 1
    while ind < X.shape[0]:
        t = X[ind, 0]
        if fix_ind >= len(fix_bounds) or t < fix_bounds[fix_ind][0]:
            test_inds.append(ind)
        else:
            if t <= fix_bounds[fix_ind][1]:
                train_inds.append(ind)
                y_train = np.vstack((y_train, stim_pos[fix_ind]))
            else:
                fix_ind += 1
        ind += 1
    
    X_train = X[train_inds, 1:]
    # y_train = y[train_inds]
    X_test = X[test_inds, 1:]
    y_test = y[test_inds]

    X_train, X_test = robust_scaler(X_train, X_test, pretrain_mode=False)
    # X_train, X_test = normalize(X_train, X_test, 'cnn', train_subjs)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    # X_test, X_val, y_test, y_val = train_test_split(
    #     X_test, y_test, test_size=0.4, random_state=42)

    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def filter_by_calib_grid(root, subj_id, calib_grid=29, manual=False, BEG_LAT=725, END_LAT=225):
    subj_dir = find_record_dir(root, subj_id)
    data_name = r'..\output_manual_recomputed_medians.csv'
    data = pd.read_csv(os.path.join(root, subj_dir, data_name), sep=',')
    data = data.drop(data[data.subj_id != int(subj_id)].index)
    data = data.reset_index()

    not_in_grid = []
    if calib_grid <= 25 and calib_grid != 13:
        not_in_grid += [5, 10, 11, 21]
    if calib_grid <= 21:
        not_in_grid += [13, 18, 26, 24]
    if calib_grid <= 15:
        not_in_grid += [9, 16, 23, 22, 19, 20]
    if calib_grid <= 13:
        not_in_grid += [4, 6, 2, 17, 0, 28]
    if calib_grid <= 5:
        not_in_grid += [14, 25, 3, 1]

    calib_inds = [i for i in range(0, 28 + 1) if i not in not_in_grid]
    if manual:
        fix_bounds = data.loc[calib_inds, ['fix_beg', 'fix_end']].values
    else:
        stim_bounds = data.loc[calib_inds, ['stim_beg', 'stim_end']].values
        fix_bounds = np.array([(x + BEG_LAT, y - END_LAT) for (x, y) in stim_bounds])

    stim_pos = data.loc[calib_inds, ['stim_pos_x', 'stim_pos_y']].values

    return fix_bounds, stim_pos

def get_fix_bounds_stim_pos(root, subj_id):
    subj_dir = find_record_dir(root, subj_id)
    data_name = r'..\output_manual_recomputed_medians.csv'
    data = pd.read_csv(os.path.join(root, subj_dir, data_name), sep=',')
    subj_data = data[data.subj_id == int(subj_id)]
    fix_bounds = subj_data[['fix_beg', 'fix_end']].values
    stim_pos = subj_data[['stim_pos_x', 'stim_pos_y']].values
    return fix_bounds, stim_pos