import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot


def plot_calib_signal(data):
    data.loc[data.ValidityLeft == 4] = np.nan
    data['Timestamp'] = data.Timestamp.astype(float)
    data.drop(data[data.Timestamp > 30000].index, inplace=True)
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
    plot([trace_hor, trace_ver], filename='signal_index')

def plot_fixations(data):
    trace_hor = go.Scatter(
        x=data.index,
        y=data['pos_x'],
        mode='lines',
        name='horizontal'
    )
    trace_ver = go.Scatter(
        x=data.index,
        y=data['pos_y'],
        mode='lines',
        name='vertical'
    )
    plot([trace_hor, trace_ver], filename='fixations')

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

def calc_and_plot_velocity(data, filename):
    # data['Timestamp'] = data.Timestamp.astype(float)
    
    # timestamp = data['Timestamp'].values

    data.drop(data[data.index > 3750].index, inplace=True)
    timestamp = data.index
    dt = np.array(np.roll(timestamp, -1, axis=0) - timestamp)[:-1]
    data['dt'] = np.concatenate(([0], dt))

    pos_x = data['GazePointXLeft'].values
    pos_y = data['GazePointYLeft'].values
    radial_pos = np.sqrt(pos_x ** 2 + pos_y ** 2)
    dp = np.array(np.roll(radial_pos, -1, axis=0) - radial_pos)[:-1]
    data['dp'] = np.concatenate(([0], dp))

    data['dv'] = data['dp'] / data['dt']
    
    print(data)

    trace = go.Scatter(
        x=data.index,
        y=data['dv'],
        mode='lines',
        name='horizontal'
    )
    plot([trace], filename=filename)

if __name__ == '__main__':
    signal_data = pd.read_csv('DOT-R51.tsv', sep='\t')
    fixations_data = pd.read_csv(
        'calib_fixations.txt',
        sep='\t', skiprows=4, header=None
    )
    pixels = fixations_data[[6, 7]].values
    degs = np.zeros(pixels.shape)
    for ind, pix in enumerate(pixels):
        degs[ind] = pix_to_deg(pix)
    # print(degs)
    fixations_data['pos_x'] = degs[:, 0]
    fixations_data['pos_y'] = degs[:, 1]

    plot_calib_signal(signal_data)
    # plot_fixations(fixations_data)

    signal_data.loc[signal_data.ValidityLeft == 4] = np.nan
    calc_and_plot_velocity(signal_data, 'signal_velocity')
    fixations_data['GazePointXLeft'] = degs[:, 0]
    fixations_data['GazePointYLeft'] = degs[:, 1]
    calc_and_plot_velocity(fixations_data, 'fixations_velocity')