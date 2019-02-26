import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from scipy.stats.stats import pearsonr


def plot_pos(data, filename, auto_open=True):
    data.drop(data[data.index > 3750].index, inplace=True)
    trace_hor = go.Scatter(
        # x=data['Timestamp'],
        x=data.index,
        y=data['GazePointXLeft'],
        mode='lines',
        name='horizontal',
    )
    trace_ver = go.Scatter(
        # x=data['Timestamp'],
        x=data.index,
        y=data['GazePointYLeft'],
        mode='lines',
        name='vertical'
    )
    plot([trace_hor, trace_ver], filename=filename, auto_open=auto_open)

def calc_radial_velocity(data):
    timestamp = data.index
    dt = np.array(np.roll(timestamp, -1, axis=0) - timestamp)[:-1]

    pos_x = data['GazePointXLeft'].values
    pos_y = data['GazePointYLeft'].values
    data['rp'] = np.sqrt(pos_x ** 2 + pos_y ** 2)
    dpx = np.array(np.roll(pos_x, -1, axis=0) - pos_x)[:-1]
    dpy = np.array(np.roll(pos_y, -1, axis=0) - pos_y)[:-1]
    drp = np.sqrt(dpx ** 2 + dpy ** 2)

    data['drv'] = np.concatenate(([0], drp / dt))

    return data

def calc_velocity(data):
    data.drop(data[data.index > 3750].index, inplace=True)
    
    timestamp = data.index
    dt = np.array(np.roll(timestamp, -1, axis=0) - timestamp)[:-1]

    pos_x = data['GazePointXLeft'].values
    pos_y = data['GazePointYLeft'].values
    radial_pos = np.sqrt(pos_x ** 2 + pos_y ** 2)
    dp = np.array(np.roll(radial_pos, -1, axis=0) - radial_pos)[:-1]

    data['dv'] = np.concatenate(([0], dp / dt))

    return data

def calc_and_plot_vel(data, filename):

    data.drop(data[data.index > 3750].index, inplace=True)
    
    data = calc_radial_velocity(data)

    print(data[data.drv > 5])

    trace = go.Scatter(
        x=data.index,
        y=data['drv'],
        mode='lines',
        name='horizontal'
    )
    plot([trace], filename=filename)

def get_calib_fixations(data):
    data = calc_radial_velocity(data)
    fixations = []
    boundaries = np.concatenate((
        [-1],
        data[data.drv > 5].index.values,
        [data.index[-1] + 1]
    ))
    LEN_THRESH = 60
    for i in range(len(boundaries) - 1):
        onset = boundaries[i] + 1
        offset = boundaries[i + 1] - 1
        if offset - onset > LEN_THRESH:
            fixations.append((onset, offset))
    return fixations

def convert_fixations_pix_to_deg(data):
    def pix_to_deg(pix):
        x, y = pix
        dist_mm = 500.
        w_mm = 374.
        h_mm = 300.
        w_pix = 1280
        h_pix = 1024
        conv = lambda data, pix, mm, dist: \
            np.arctan((data - pix/2.) * mm/pix / dist) / np.pi * 180.
        return conv(x, w_pix, w_mm, dist_mm), \
            -conv(y, h_pix, h_mm, dist_mm)
    def convert(pixs):
        degs = np.zeros(pixs.shape)
        for ind, pix in enumerate(pixs):
            degs[ind] = pix_to_deg(pix)
        return degs

    pos_degs = convert(data[[6, 7]].values)
    tar_degs = convert(data[[0, 1]].values)

    data['GazePointXLeft'] = pos_degs[:, 0]
    data['GazePointYLeft'] = pos_degs[:, 1]
    
    data['tar_x'] = tar_degs[:, 0]
    data['tar_y'] = tar_degs[:, 1]

    return data

def find_best_corr(sig_data, fix_data, fix_pos, start=0):
    sig_data = calc_radial_velocity(sig_data)
    fix_data = calc_radial_velocity(fix_data)

    on, off = fix_pos
    fix_len = off - on + 1

    best_corr = 0.
    best_on = best_off = 0
    N_CORR = 30
    for b in range(start, len(sig_data.index) - fix_len):
        corr, _ = pearsonr(
            # TODO: attemp to overcome hole in the signal
            sig_data[b: b + N_CORR]['rp'].values,
            fix_data[on: on + N_CORR]['rp'].values
        )
        if corr > best_corr:
            best_corr = corr
            best_on = b

    best_corr = 0.
    HOW_FAR = 2*fix_len
    for e in range(best_on, min(best_on + HOW_FAR, len(sig_data.index))):
        corr, _ = pearsonr(
            # TODO: attemp to overcome hole in the signal
            sig_data[e: e - N_CORR: -1]['rp'].values,
            fix_data[off: off - N_CORR: -1]['rp'].values
        )
        if corr > best_corr:
            best_corr = corr
            best_off = e
    
    return best_corr, best_on, best_off

def plot_subj_calib(subj_root, plot=True):
    print('Matching calibration signals for: ', subj_root)
    tsv_file = None
    for filename in os.listdir(subj_root):
        if filename.endswith('.tsv'):
            tsv_file = filename
    
    tsv_path = os.path.join(subj_root, tsv_file)
    signal_data = pd.read_csv(tsv_path, sep='\t')
    
    txt_path = os.path.join(subj_root, 'calib_fixations.txt')
    fixations_data = pd.read_csv(
        txt_path,
        sep='\t', skiprows=4, header=None
    )
    
    signal_data.loc[signal_data.ValidityLeft == 4] = np.nan
    fixations_data = convert_fixations_pix_to_deg(fixations_data)

    html_dir = os.path.join(subj_root, 'html')
    if not os.path.exists(html_dir):
        os.mkdir(html_dir)

    fix_sig_map = []
    for ind, fixation in enumerate(get_calib_fixations(fixations_data)):
        corr, b, e = find_best_corr(signal_data, fixations_data, fixation)
        on, off = fixation
        fix_sig_map.append((b, e, on, off, corr))
        
        sig_filename = os.path.join(html_dir, str(ind) + '_sig_corr')
        fix_filename = os.path.join(html_dir, str(ind) + '_fix_corr')

        if plot:
            plot_pos(signal_data[b: e + 1], sig_filename, False)
            plot_pos(fixations_data[on: off + 1], fix_filename, False)
    
    calib_data = pd.DataFrame(
        np.nan,
        index=signal_data.index, columns=signal_data.columns
    )
    target_data = pd.DataFrame(
        np.nan,
        index=signal_data.index, columns=['tar_x', 'tar_y']
    )

    max_disp = 0
    for m in fix_sig_map:
        b, e, on, off, _ = m
        calib_data[b: e + 1] = signal_data[b: e + 1]
        target_data.iloc[b: e + 1, :] = fixations_data[['tar_x', 'tar_y']].values[on]
        disp = signal_data[b: e + 1]['rp'].dropna().max() - signal_data[b: e + 1]['rp'].dropna().min()
        if disp > max_disp:
            max_disp = disp
    calib_data = calib_data.join(target_data)

    data_path = os.path.join(subj_root, 'FullSignal.csv')
    calib_data.to_csv(data_path, sep='\t', na_rep=np.nan)

    print(fix_sig_map)
    print('subj_root disp:', disp)

def plot_dataset(root, subj_ids=None):
    for dirname in os.listdir(root):
        if subj_ids is not None and dirname not in subj_ids:
            continue
        subj_root = os.path.join(root, dirname)
        plot_subj_calib(subj_root, False)


if __name__ == '__main__':
    plot_dataset(sys.argv[1], ['Record 15'])