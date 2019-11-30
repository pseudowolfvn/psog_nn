import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from ml.load_data import get_subj_data, reshape_into_grid, filter_outliers
from ml.utils import robust_scaler
from preproc.psog import PSOG
from utils.metrics import calc_acc
from utils.utils import find_record_dir, find_filename


def get_stim_pos_subj_data(root, subj_id):
    subj_dir = find_record_dir(root, subj_id)
    subj_root = os.path.join(root, subj_dir)

    data_name = find_filename(
        subj_root, 'StimulusPosition.tsv',
        beg='Stimulus', end='.tsv'
    )

    data = pd.read_csv(os.path.join(subj_root, data_name), sep='\t')

    mask = ['Timestamp', 'Parameter1', 'Parameter2']
    return data[data.EventType == 3][mask][1:].values

def get_subj_calib_data(root, subj_id, data_id):
    subj_dir = find_record_dir(root, subj_id)
    subj_root = os.path.join(root, subj_dir)

    data_name = find_filename(
        subj_root, '',
        beg='', end='randn_v3_pred.csv'
    )
    data_path = os.path.join(subj_root, data_name)
    data = pd.read_csv(data_path, sep='\t')

    matlab_name = find_filename(subj_root, '', beg='Recording_', end='Velocity.mat')
    matlab_path = os.path.join(subj_root, matlab_name)
    matlab_data = loadmat(matlab_path)

    data = filter_outliers(data, matlab_data)

    stim_pos = get_stim_pos_subj_data(root, subj_id)

    data['stim_pos_x'] = np.nan
    data['stim_pos_y'] = np.nan

    for i in range(len(stim_pos) - 1):
    # for i in range(1):
        beg = stim_pos[i][0]
        end = stim_pos[i + 1][0]

        stim_pos_x = stim_pos[i][1]
        stim_pos_y = stim_pos[i][2]

        time_mask = (data.time >= beg) & (data.time < end)
        data.loc[time_mask, 'stim_pos_x'] = stim_pos_x
        data.loc[time_mask, 'stim_pos_y'] = stim_pos_y
    
    return data

def get_fix_bounds_stim_pos(root, subj_id):
    subj_dir = find_record_dir(root, subj_id)
    subj_root = os.path.join(root, subj_dir)

    # data_name = r'..\output_manual_recomputed_medians.csv'
    # data_name = r'..\output_dbscan_medians.csv'
    data_name = r'..\output_dbscan_medians_val.csv'
    data = pd.read_csv(os.path.join(subj_root, data_name), sep=',')

    subj_data = data[data.subj_id == int(subj_id)]

    fix_bounds = subj_data[['fix_beg', 'fix_end']].values
    stim_pos = subj_data[['stim_pos_x', 'stim_pos_y']].values

    return fix_bounds, stim_pos

def make_calib_grid_specific_data(data_id='randn', calib_grid=29, parser_mode='gt_vog'):
    def f(root, subj_id, arch, train_subjs):
        return get_specific_data(
            root, subj_id, arch,
            train_subjs=train_subjs,
            data_id=data_id,
            calib_grid=calib_grid,
            parser_mode=parser_mode
        )
    return f

def get_specific_data(root, subj_id, arch, train_subjs=None, data_id='randn', calib_grid=29, parser_mode=None):
    # print('DEBUG, getting data for:', subj_id, train_subjs, data_id, calib_grid, manual)
    print('DEBUG, getting data for:', subj_id, parser_mode)

    data = get_subj_calib_data(root, subj_id, data_id=data_id)
    stim_pos = get_stim_pos_subj_data(root, subj_id)

    fix_bounds, _ = get_fix_bounds_stim_pos(root, subj_id)
    if parser_mode == 'gt_vog':
        data = ground_truth_vog(data, fix_bounds)
    elif parser_mode == 'blind_baseline':
        data = blind_baseline(data, stim_pos)
    elif parser_mode == 'blind_temporal':
        data = blind_temporal(data, stim_pos)
    elif parser_mode == 'all_fix_uncalib_psog':
        data = all_fix_uncalibrated_psog(data, stim_pos)
    elif parser_mode == 'longest_fix_uncalib_psog':
        data = longest_fix_uncalibrated_psog(data, stim_pos)
    elif parser_mode == 'temporal_stable_regions':
        data = temporal_with_stable_regions(data, stim_pos)
    else:
        print('ERROR: Unknown parsing mode:', parser_mode)
        exit()

    data = mark_val_set(data, fix_bounds, stim_pos)
    data = filter_by_calib_grid(data, stim_pos, calib_grid=calib_grid)

    subj_dir = find_record_dir(root, subj_id)
    subj_root = os.path.join(root, subj_dir)
    data.to_csv(os.path.join(subj_root, 'DATA_' + parser_mode + '.csv'), sep='\t', index=False)

    return get_train_calib_data(data, arch)

def temporal_with_stable_regions(data, stim_pos, BEG_DEL=735, END_DEL=-155):
    stim_pos = filter_by_phase(stim_pos)

    data['calib_fix'] = 0
    for i in range(len(stim_pos) - 1):
    # for i in range(1):
        beg = stim_pos[i][0]
        end = stim_pos[i + 1][0]

        time_mask = (data.time >= beg + BEG_DEL) & (data.time <= end + END_DEL)
        data.loc[time_mask, 'calib_fix'] = data.loc[time_mask, 'prediction']

    return data

def all_fix_uncalibrated_psog(data, stim_pos):
    stim_pos = filter_by_phase(stim_pos)
    beg = stim_pos[0][0]
    end = stim_pos[-1][0]

    all_fix_mask = (data.time >= beg) & \
        (data.time <= end) & \
        (data.prediction == 1)

    data['calib_fix'] = 0
    data.loc[all_fix_mask, 'calib_fix'] = 1

    return data

def longest_fix_uncalibrated_psog(data, stim_pos):
    def get_stim_fix_mask(data, beg, end):
        return (data.time >= beg) & \
            (data.time <= end) & \
            (data.prediction == 1)

    def get_longest_fix(bounds):
        best = bounds[0]
        for b in bounds:
            if (best[1] - best[0]) < (b[1] - b[0]):
                best = b
        return best

    stim_pos = filter_by_phase(stim_pos)

    data['calib_fix'] = 0
    for i in range(len(stim_pos) - 1):
    # for i in range(1):
        beg = stim_pos[i][0]
        end = stim_pos[i + 1][0]

        all_stim_fix_mask = get_stim_fix_mask(data, beg, end)

        all_stim_fix = get_fix_bounds_from_timestamps(
            data.loc[all_stim_fix_mask].time.values
        )

        # no fixation at the calibration target detected
        if len(all_stim_fix) == 0:
            continue

        longest_stim_fix = get_longest_fix(all_stim_fix)
        beg, end = longest_stim_fix
        longest_stim_fix_mask = get_stim_fix_mask(data, beg, end)
        
        data.loc[longest_stim_fix_mask, 'calib_fix'] = \
            data.loc[longest_stim_fix_mask, 'prediction']

    return data


def blind_baseline(data, stim_pos, LAT=225):
    return blind_temporal(data, stim_pos, BEG_DEL=LAT, END_DEL=-LAT)

def blind_temporal(data, stim_pos, BEG_DEL=735, END_DEL=-155):
    stim_pos = filter_by_phase(stim_pos)

    data['calib_fix'] = 0
    for i in range(len(stim_pos) - 1):
    # for i in range(1):
        beg = stim_pos[i][0]
        end = stim_pos[i + 1][0]

        time_mask = (data.time >= beg + BEG_DEL) & (data.time <= end + END_DEL)
        data.loc[time_mask, 'calib_fix'] = 1

    return data

def ground_truth_vog(data, fix_bounds):
    fix_bounds = filter_by_phase(fix_bounds)

    data['calib_fix'] = 0
    for fix in fix_bounds:
        beg, end = fix

        time_mask = (data.time >= beg) & (data.time <= end)
        data.loc[time_mask, 'calib_fix'] = 1

    return data

def get_fix_bounds_from_timestamps(ts):
    bounds = []
    inside = False
    b = e = None
    for i in range(1, len(ts)):
        if (ts[i] - ts[i - 1]) < 10.:
            if not inside:
                b = ts[i - 1]
                inside = True
        else:
            if inside:
                e = ts[i - 1]
                bounds.append((b, e))
                inside = False
    if inside:
        e = ts[-1]
        bounds.append((b, e))
    return bounds

def get_train_calib_data(data, arch):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        split_data_train_val_calib_test(data, with_time=True)

    # fixation_timestamps_sanity_check(X_train[:, 0])
    X_train = X_train[:, 1:]
    X_val = X_val[:, 1:]
    X_test = X_test[:, 1:]

    X_train, [X_val, X_test] = robust_scaler(X_train, X_to_scale=[X_val, X_test])

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=42)
    
    if arch == 'cnn':
        X_train, X_val, X_test = reshape_into_grid(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
    # return test set as validation
    # useful for debugging purposes
    # return X_train, X_test, X_test, y_train, y_test, y_test

def split_data_train_val_calib_test(data, with_shifts=False, with_time=False):
    def compliance_report(i, y_gt, y_stim):
        print(
            'Target {:-3d}: ({:-6.2f}, {:-6.2f})'.format(i, *y_stim[0]),
            'Err: {:.3f}'.format(
                calc_acc(
                    y_gt,
                    y_stim
                )
            )
        )

    X_cols = (['time'] if with_time else []) + \
        PSOG().get_names() + \
        (['sh_hor', 'sh_ver'] if with_shifts else [])

    train_val_mask = (data.calib_fix == 1) | (data.calib_fix == 2)
    # because the fixation can be longer than a calibration target
    # it can overlap with the next one!
    # so we will assign the calibration position according to the first
    # timestamp of every fixation instead of directly using 
    # data.stim_pos_x and data.stim_pos_y
    fix_ts = data.loc[train_val_mask, 'time'].values
    train_val_fix = get_fix_bounds_from_timestamps(fix_ts)
    data.to_csv('debug_calib_data.csv', sep='\t')

    X_train = []
    y_train = []

    X_val = []
    y_val = []

    for i, fix in enumerate(train_val_fix):
        beg, end = fix
        time_mask = (data.time >= beg) & (data.time <= end)

        stim_pos = data[data.time == beg][['stim_pos_x', 'stim_pos_y']].values
        stim_len = data.loc[time_mask].shape[0]

        X_temp = data.loc[time_mask, X_cols].values
        y_temp = np.tile(stim_pos, (stim_len, 1))

        # compliance_report(i, data.loc[time_mask][['pos_x', 'pos_y']].values, y_temp)

        if data[data.time == beg].calib_fix.values[0] == 2:
            X_val.extend(X_temp)
            y_val.extend(y_temp)
        else:
            X_train.extend(X_temp)
            y_train.extend(y_temp)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    test_mask = data.calib_fix == 0
    X_test = data.loc[test_mask, X_cols].values
    y_test = data.loc[test_mask, ['pos_x', 'pos_y']].values

    return X_train, X_val, X_test, y_train, y_val, y_test

def filter_by_calib_grid(data, stim_pos, calib_grid=29):
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

    for i in not_in_grid:
        beg = stim_pos[i][0]
        end = stim_pos[i + 1][0]

        time_mask = (data.time >= beg) & (data.time < end)
        data.loc[time_mask, 'calib_fix'] = 0

    return data

def mark_val_set(data, fix_bounds, stim_pos):
    def mark_best_val_by_compliance(data, fix_bounds, stim_pos, inds):
        def compute_compliance(data, tar_fix, tar_pos):
            b, e = tar_fix
            fix_mask = (data.time >= b) & (data.time <= e)
            fix_pos = data.loc[fix_mask, ['pos_x', 'pos_y']].values
            compl = calc_acc(
                np.array([np.median(fix_pos, 0)]),
                np.array([tar_pos])
            )
            print('VAL, compliance:', tar_fix, tar_pos, compl)
            return compl

        best = None
        best_ind = None
        for i in inds:
            curr = compute_compliance(data, fix_bounds[i], stim_pos[i, -2:])
            if best is None or curr < best:
                best = curr
                best_ind = i

        print('VAL, the best target for validation:', stim_pos[best_ind, -2:])
        b, e = fix_bounds[best_ind]
        val_mask = (data.time >= b) & (data.time <= e)
        data.loc[val_mask, 'calib_fix'] = 2

        return data

    val_inds_per_quadrant = [
        # I Quadrant
        [31, 33, 54, 72],
        # II Quadrant
        [30, 60, 70, 71],
        # III Quadrant
        [36, 61, 63, 67],
        # IV Quadrant
        [57, 58, 68, 75]
    ]
    
    for q, inds in enumerate(val_inds_per_quadrant):
        print('VAL, getting the best target for quadrant: ', q + 1)
        data = mark_best_val_by_compliance(data, fix_bounds, stim_pos, inds)
    
    return data

def filter_by_phase(data, ph='1'):
    phases = {
        '1': (0, 29),
        '2': (29, 78),
        '3': (78, 174)
    }
    id_from = phases[ph][0]
    id_to = phases[ph][1] + 1
    return data[id_from: id_to]

if __name__ == '__main__':
    import sys
    root = sys.argv[1]
    subj_id = str(sys.argv[2])

    f = make_calib_grid_specific_data(parser_mode='all_fix_uncalib_psog')

    X_train, X_val, X_test, y_train, y_val, y_test = f(root, subj_id, 'mlp', train_subjs=None)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    # zero_pos_train = (np.abs(y_train[:, 0]) < 0.5) & (np.abs(y_train[:, 1]) < 0.5)
    # zero_pos_test = (np.abs(y_test[:, 0]) < 0.5) & (np.abs(y_test[:, 1]) < 0.5)
    # train_stats = np.median(X_train[zero_pos_train], axis=0)
    # test_stats = np.median(X_test[zero_pos_test], axis=0)
    # # diff_stats = (train_stats - test_stats) / train_stats
    # stats = np.vstack((X_train[zero_pos_train], X_test[zero_pos_test]))
    # diff_stats = np.std(stats, axis=0)
    # for d in diff_stats:
    #     print('{:-6.2f}'.format(d))
    # print(np.max(diff_stats))
    exit()

    train_ts = get_fix_bounds_from_timestamps(X_train[:, 0])
    val_ts = get_fix_bounds_from_timestamps(X_val[:, 0])

    # i = 0
    # for ts in val_ts:
    #     b, e = ts
    #     while X_val[i, 0] != b:
    #         i += 1
    #     print(b, X_val[i, 0], y_val[i, :])
    #     while X_val[i, 0] != e:
    #         i += 1
    #     print(e, X_val[i, 0], y_val[i, :])

    def plot_stats(X_train, X_val, X_test):
        import plotly.graph_objs as go
        from plotly.offline import plot
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=3, cols=X_train.shape[1])

        row = 1
        for X, name in zip([X_train, X_val, X_test], ['train', 'val', 'test']):
            N = X.shape[1]
            col = 1
            for i in range(N):
                trace = go.Histogram(
                    x=X[:, i],
                    name=name + '_' + str(i),
                    histnorm='probability'
                )
                print('Append: ', row, col)
                fig.append_trace(trace, row, col)
                col += 1

            row += 1

        fig.update_xaxes(range=[-2., 2.])
        plot(fig)

    plot_stats(X_train, X_val, X_test)