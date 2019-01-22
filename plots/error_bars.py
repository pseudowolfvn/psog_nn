import os
import numpy as np
from sklearn.externals import joblib
import pandas as pd
from plotly import tools
from plotly.offline import plot
import plotly.graph_objs as go

from color_map import plotly_color_map

FONT_SIZE = 18

data = pd.read_csv('DataSet1.csv', sep=',')

group = {
    'means': [
        data[data['ArchAppr'] == 1].mean()['acc'],
        data[data['ArchAppr'] == 2].mean()['acc'],
        data[data['ArchAppr'] == 3].mean()['acc'],
        data[data['ArchAppr'] == 4].mean()['acc'],
    ],
    'stds': [
        data[data['ArchAppr'] == 1].std()['acc'],
        data[data['ArchAppr'] == 2].std()['acc'],
        data[data['ArchAppr'] == 3].std()['acc'],
        data[data['ArchAppr'] == 4].std()['acc'],
    ]
}

cm = plotly_color_map(['1', '2', '3', '4'])
labels = ['MLP: fine-tune', 'MLP: from scratch', 'CNN: fine-tune', 'CNN: from scratch']

def get_trace_for_group(n):
    mean = group['means'][n]
    std = group['stds'][n]
    return go.Scatter(
            x=[labels[n]],
            y=[mean],
            error_y=dict(
                type='data',
                array=[std],
                color=cm[str(n + 1)],
                visible=True
            ),
            mode='markers+text',
            marker={'color': cm[str(n + 1)]},
            text=[str(round(mean, 4)) + u' \u00B1 ' + str(round(std, 4))],
            textposition='middle right',
            textfont={
                'size': FONT_SIZE,
            }
        )

fig = go.Figure(
    data=[get_trace_for_group(0), 
        get_trace_for_group(1), 
        get_trace_for_group(2), 
        get_trace_for_group(3),
    ],
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
    fig, filename='dataset_1_paper'
)

