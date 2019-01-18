import os
import numpy as np
from sklearn.externals import joblib
from plotly import tools
from plotly.offline import plot
import plotly.graph_objs as go

from color_map import plotly_color_map

CNN_MODEL = '(4, 4, 4, 20)'

ROWS = 2
COLS = 4

def accumulate_data(path, subjs, data):
    temp = joblib.load(path)
    subjs = subjs + temp['subjs']
    data = {**data, **temp}
    return subjs, data

data = {}
subjs = []
subjs, data = accumulate_data(
    'finetune_cnn_' + CNN_MODEL + '_[\'1\', \'2\', \'3\', \'4\'].pkl', subjs, data)
subjs, data = accumulate_data(
    'finetune_cnn_' + CNN_MODEL + '_[\'5\', \'6\', \'7\', \'8\'].pkl', subjs, data)
subjs, data = accumulate_data(
    'finetune_cnn_' + CNN_MODEL + '_[\'10\', \'11\', \'12\', \'13\'].pkl', subjs, data)
subjs, data = accumulate_data(
    'finetune_cnn_' + CNN_MODEL + '_[\'14\', \'15\', \'16\', \'17\'].pkl', subjs, data)
subjs, data = accumulate_data(
    'finetune_cnn_' + CNN_MODEL + '_[\'18\', \'19\', \'20\'].pkl', subjs, data)
subjs, data = accumulate_data(
    'finetune_cnn_' + CNN_MODEL + '_[\'21\', \'22\', \'23\'].pkl', subjs, data)

subjs = subjs[:8]
#subjs = subjs[8:16]
#subjs = subjs[16:]

fig = tools.make_subplots(rows=ROWS, cols=COLS,
    subplot_titles=['Subject ' + subj for subj in subjs])

cm = plotly_color_map(['ft', 'ft_fc', 'scr'])

ind = 0
legend = True

for subj in subjs:
    if subj not in data:
        ind += 1
        continue

    ft_time = round(np.mean(data[subj]['ft']['time']), 2)
    ft = go.Box(y=data[subj]['ft']['data'], name='Finetune',
        legendgroup='ft', marker={'color': cm['ft']}, boxmean='sd', showlegend=legend)

    ft_fc_time = round(np.mean(data[subj]['ft_fc']['time']), 2)
    ft_fc = go.Box(y=data[subj]['ft_fc']['data'], name='Finetune FC',
        legendgroup='ft_fc', marker={'color': cm['ft_fc']}, boxmean='sd', showlegend=legend)

    scr_time = round(np.mean(data[subj]['scr']['time']), 2)
    scr = go.Box(y=data[subj]['scr']['data'], name='Scratch',
        legendgroup='scr', marker={'color': cm['scr']}, boxmean='sd', showlegend=legend)

    fig.append_trace(ft, ind // COLS + 1, ind % COLS + 1)
    fig.append_trace(ft_fc, ind // COLS + 1, ind % COLS + 1)
    fig.append_trace(scr, ind // COLS + 1, ind % COLS + 1)

    ind += 1
    legend = False

for i in range(1, ROWS*COLS + 1):
    fig['layout'].update(title='CNN: Conv2D(4)x4 -> Dense(20)x4')
    fig['layout']['xaxis' + str(i)].update(title='Approach')
    fig['layout']['yaxis' + str(i)].update(title='Spatial accuracy')
    fig['layout']['xaxis' + str(i)].update(showticklabels=False)

plot(
    fig, filename = 'lp_cnn_1:8'
)

