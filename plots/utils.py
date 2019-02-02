from itertools import cycle
import os

import pandas as pd
from sklearn.externals import joblib

from utils.utils import get_arch


def get_module_prefix():
    return r'.\plots'

def plotly_color_map(names):
    # From https://stackoverflow.com/a/44727682
    plotly_colors = cycle(['#1f77b4',  # muted blue
                           '#ff7f0e',  # safety orange
                           '#2ca02c',  # cooked asparagus green
                           '#d62728',  # brick red
                           '#9467bd',  # muted purple
                           '#8c564b',  # chestnut brown
                           '#e377c2',  # raspberry yogurt pink
                           '#7f7f7f',  # middle gray
                           '#bcbd22',  # curry yellow-green
                           '#17becf'  # blue-teal
                           ])

    return dict(zip(names, plotly_colors))

def accumulate_data(path, data):
    temp = joblib.load(path)
    for k, v in temp.items():
        if k == 'subjs' and 'subjs' in data:
            data[k].extend(v)
        else:
            data[k] = v
    return data

def extract_params(name):
    opened = name.find('(')
    closed = name.find(')')
    return tuple(map(int, name[opened + 1: closed].split(', ')))

def load_data(data_root, arch, setup):
    data = {}
    arch_params = ()
    for filename in os.listdir(data_root):
        params = extract_params(filename)
        interested = (arch == get_arch(params) and
            setup in filename and 
            filename.endswith('.pkl'))
        if not interested:
            continue
        data = accumulate_data(
            os.path.join(data_root, filename),
            data
        )
    return data, arch_params

def calc_stats(data):
    stats = {
        'means': [
            data[data['Group'] == 1].mean()['acc'],
            data[data['Group'] == 2].mean()['acc'],
            data[data['Group'] == 3].mean()['acc'],
            data[data['Group'] == 4].mean()['acc'],
        ],
        'stds': [
            data[data['Group'] == 1].std()['acc'],
            data[data['Group'] == 2].std()['acc'],
            data[data['Group'] == 3].std()['acc'],
            data[data['Group'] == 4].std()['acc'],
        ]
    }
    return stats

def accumulate_group(groups, data, subj, arch, appr):
    subj_data = data[arch][str(subj)]

    subj_arch = 1 if arch == 'mlp' else 2
    subj_appr = 1 if appr.startswith('ft') else 2
    group = subj_appr if subj_arch == 1 else subj_appr + subj_arch

    for ind, acc in enumerate(subj_data[appr]['data']):
        groups = groups.append({
            'subj': subj,
            'arch': subj_arch,
            'appr': subj_appr,
            'Group': group,
            'rep': ind + 1,
            'acc': acc
        }, ignore_index=True)
    return groups

def convert_to_groups(mlp_data, cnn_data):
    groups = pd.DataFrame(columns=['subj', 'arch', 'appr', 'Group', 'rep', 'acc'])

    data = {
        'mlp': mlp_data,
        'cnn': cnn_data
    }
    subjs = mlp_data['subjs']
    for subj in subjs:
        groups = accumulate_group(groups, data, subj, 'mlp', 'ft') 
        groups = accumulate_group(groups, data, subj, 'mlp', 'scr')

        groups = accumulate_group(groups, data, subj, 'cnn', 'ft_fc') 
        groups = accumulate_group(groups, data, subj, 'cnn', 'scr')

    groups['subj'] = groups['subj'].astype(int)
    groups['arch'] = groups['arch'].astype(int)
    groups['appr'] = groups['appr'].astype(int)
    groups['rep'] = groups['rep'].astype(int)

    return groups
