import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from scipy.stats.stats import pearsonr


def plot_pos(data, filename):
    data.drop(data[data.index > 3750].index, inplace=True)
    trace_hor = go.Scatter(
        # x=data['Timestamp'],
        x=data.index,
        y=data['GazePointXLeft'],
        mode='lines',
        name='horizontal'
    )
    trace_ver = go.Scatter(
        # x=data['Timestamp'],
        x=data.index,
        y=data['GazePointYLeft'],
        mode='lines',
        name='vertical'
    )
    plot([trace_hor, trace_ver], filename=filename)

def calc_radial_velocity(data):
    timestamp = data.index
    dt = np.array(np.roll(timestamp, -1, axis=0) - timestamp)[:-1]

    pos_x = data['GazePointXLeft'].values
    pos_y = data['GazePointYLeft'].values
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

    print(data[data.dv > 5])

    trace = go.Scatter(
        x=data.index,
        y=data['dv'],
        mode='lines',
        name='horizontal'
    )
    plot([trace], filename=filename)

def get_calib_fixations(data):
    data = calc_radial_velocity(data)
    fixations = []
    boundaries = np.concatenate((
        [-1],
        data[data.dv > 5].index.values,
        [data.index[-1] + 1]
    ))
    LEN_THRESH = 30
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
    
    pixels = data[[6, 7]].values
    degs = np.zeros(pixels.shape)
    for ind, pix in enumerate(pixels):
        degs[ind] = pix_to_deg(pix)
    
    data['GazePointXLeft'] = degs[:, 0]
    data['GazePointYLeft'] = degs[:, 1]
    
    return data

def plots(sig_data, fix_data):
    plot_pos(sig_data, 'signal_pos')
    plot_pos(fix_data, 'fixations_pos')

    calc_and_plot_vel(sig_data, 'signal_velocity')
    calc_and_plot_vel(fix_data, 'fixations_velocity')

def find_best_corr(sig_data, fix_data, fix_pos):
    sig_data = calc_velocity(sig_data)
    fix_data = calc_velocity(fix_data)

    on, off = fix_pos
    fix_len = off - on + 1

    best_corr = 0.
    best_shift = 0
    for b in range(len(sig_data.index) - fix_len):
        corr, _ = pearsonr(
            sig_data[b: b + fix_len]['dv'].values,
            fix_data[on: off]['dv'].values
        )
        if corr > best_corr:
            best_corr = corr
            best_shift = b
    
    return best_shift, best_shift + fix_len


if __name__ == '__main__':
    signal_data = pd.read_csv('DOT-R51.tsv', sep='\t')
    signal_data.loc[signal_data.ValidityLeft == 4] = np.nan

    fixations_data = pd.read_csv(
        'calib_fixations.txt',
        sep='\t', skiprows=4, header=None
    )
    fixations_data = convert_fixations_pix_to_deg(fixations_data)

    for fixation in get_calib_fixations(fixations_data):
        b, e = find_best_corr(signal_data, fixations_data, fixation)
        print(b, e)
        break