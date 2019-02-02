import os
import sys

import numpy as np
from sklearn.externals import joblib
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import plot

from .utils import load_data, plotly_color_map, get_module_prefix
from utils.utils import get_arch

def get_arch_title(params):
    L_conv, D, L_fc, N = params 
    title = 'Dense({})x{}'.format(L_fc, N)
    if get_arch(params) == 'cnn':
        title = 'CNN: Conv2D({})x{} -> '.format(L_conv, D) + title
    return title

def plot_subjs(data, subjs, arch_params, setup):
    ROWS = 2
    COLS = 4 

    fig = tools.make_subplots(rows=ROWS, cols=COLS,
        subplot_titles=['Subject ' + subj for subj in subjs])

    cm = plotly_color_map(['ft', 'ft_fc', 'scr'])

    arch = get_arch(arch_params)
    legend = True
    for ind, subj in enumerate(subjs):
        # ft_time = round(np.mean(data[subj]['ft']['time']), 2)
        ft = go.Box(y=data[subj]['ft']['data'], name='Finetune',
            legendgroup='ft', marker={'color': cm['ft']}, boxmean='sd', showlegend=legend)
        fig.append_trace(ft, ind // COLS + 1, ind % COLS + 1)

        if arch == 'cnn':
            # ft_fc_time = round(np.mean(data[subj]['ft_fc']['time']), 2)
            ft_fc = go.Box(y=data[subj]['ft_fc']['data'], name='Finetune FC',
                legendgroup='ft_fc', marker={'color': cm['ft_fc']}, boxmean='sd', showlegend=legend)
            fig.append_trace(ft_fc, ind // COLS + 1, ind % COLS + 1)

        # scr_time = round(np.mean(data[subj]['scr']['time']), 2)
        scr = go.Box(y=data[subj]['scr']['data'], name='Scratch',
            legendgroup='scr', marker={'color': cm['scr']}, boxmean='sd', showlegend=legend)
        fig.append_trace(scr, ind // COLS + 1, ind % COLS + 1)
        
        legend = False

    for i in range(1, ROWS*COLS + 1):
        fig['layout'].update(title=get_arch_title(arch_params))
        fig['layout']['xaxis' + str(i)].update(title='Approach')
        fig['layout']['yaxis' + str(i)].update(title='Spatial accuracy')
        fig['layout']['xaxis' + str(i)].update(showticklabels=False)

    plot_dir = os.path.join(get_module_prefix(), 'boxplots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(
        plot_dir,
        setup + '_' + arch + '_' + str(subjs[0]) + ':' + str(subjs[-1])
    )
    plot(
        fig,
        filename=plot_path
    )

def plot_setup(root, arch, setup):
    data, arch_params = load_data(root, arch, setup)
    subjs = data['subjs']
    # be aware that it will work only if subject's id is an integer!
    subjs.sort(key=int)
    plot_subjs(data, subjs[:8], arch_params, setup)
    plot_subjs(data, subjs[8:16], arch_params, setup)
    plot_subjs(data, subjs[16:], arch_params, setup)    

def plot_study(root, archs, setups):
    for arch in archs:
        for setup in setups:
            plot_setup(root, arch, setup)

if __name__ == '__main__':
    root = sys.argv[1]
    plot_study(root, ['mlp', 'cnn'], ['lp', 'hp'])