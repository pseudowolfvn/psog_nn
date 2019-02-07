""" Plot per setup resulting spatial accuracy error bars.
"""
import os
import sys

import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

from .utils import calc_stats, convert_to_groups, \
    get_module_prefix, load_data, plotly_color_map


FONT_SIZE = 18

def get_trace_for_group(mean, std, color, label):
    return go.Scatter(
        x=[label],
        y=[mean],
        error_y=dict(
            type='data',
            array=[std],
            color=color,
            visible=True
        ),
        mode='markers+text',
        marker={'color': color},
        text=[str(round(mean, 4)) + u' \u00B1 ' + str(round(std, 4))],
        textposition='middle right',
        textfont={
            'size': FONT_SIZE,
        }
    )

def load_groups_data(root, setup):
    data_path = os.path.join(root, setup + '.csv')
    if not os.path.exists(data_path):
        mlp_data, _ = load_data(root, 'mlp', setup)
        cnn_data, _ = load_data(root, 'cnn', setup)
        data = convert_to_groups(mlp_data, cnn_data)
        data.to_csv(data_path, sep=',')
    else:
        data = pd.read_csv(data_path, sep=',')
    return data

def plot_setup(root, setup):
    data = load_groups_data(root, setup)
    stats = calc_stats(data)

    cm = plotly_color_map(['1', '2', '3', '4'])
    labels = [
        'MLP: fine-tune',
        'MLP: from scratch',
        'CNN: fine-tune',
        'CNN: from scratch'
    ]

    fig = go.Figure(
        data=[get_trace_for_group(
            stats['means'][i],
            stats['stds'][i],
            cm[str(i + 1)],
            labels[i]) for i in range(4)],
        layout=go.Layout(
            width=960,
            height=480,
            xaxis={
                'showgrid': False,
                'range': [-0.5, 3.8],
                'tickfont': {
                    'size': FONT_SIZE
                }
            },
            yaxis={
                'dtick': 0.05,
                'tickfont': {
                    'size': FONT_SIZE
                }
            }
        )
    )

    plot_dir = os.path.join(get_module_prefix(), 'error_bars')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(
        plot_dir,
        setup + '_error_bars'
    )
    plot(
        fig,
        filename=plot_path
    )

def plot_error_bars(root, setups):
    for setup in setups:
        plot_setup(root, setup)

if __name__ == '__main__':
    plot_error_bars(sys.argv[1], ['lp', 'hp'])
