import os
import sys

import numpy as np
import pandas as pd
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.externals import joblib

from .utils import plotly_color_map, calc_stats, load_data, convert_to_groups

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
    labels = ['MLP: fine-tune',
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
                xaxis = {
                    'showgrid': False,
                    'range': [-0.5,3.8],
                    'tickfont': {
                        'size': FONT_SIZE
                    }
                },
                yaxis = {
                    'dtick': 0.05,
                    'tickfont': {
                        'size': FONT_SIZE
                    }
                }
            )
    )

    plot(
        fig, filename=setup + '_error_bars'
    )

if __name__ == '__main__':
    root = sys.argv[1]
    plot_setup(root, 'lp')
    plot_setup(root, 'hp')
