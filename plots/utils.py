""" Plots utility functions.
"""
from itertools import cycle
import os

import pandas as pd
from sklearn.externals import joblib

from utils.utils import get_arch


def get_module_prefix():
    """Get current module prefix.
    
    Returns:
        A string with module prefix.
    """
    return 'plots'

def plotly_color_map(names):
    """Map provided names to cyclic list of strings with
        hex values of colors used in Plotly's default color scheme.
    
    Args:
        names: A list of strings with names.

    Returns:
        A dict with names mapped to colors.
    """
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
    """Accumulate several dicts with partial results into one provided dict.

    Args:
        path: A string with full path to partial results.
        data: A dict to accumulate results in.
    
    Returns:
        A dict with accumulated data.
    """
    temp = joblib.load(path)
    for k, v in temp.items():
        if k == 'subjs' and 'subjs' in data:
            data[k].extend(v)
        else:
            data[k] = v
    return data

def extract_params(name):
    """Extract neural network parameters from partial results filename.

    Args:
        name: A string with filename.

    Returns:
        A tuple following the format described in ml.model.build_model().
    """
    opened = name.find('(')
    closed = name.find(')')
    return tuple(map(int, name[opened + 1: closed].split(', ')))

def load_data(data_root, arch, setup):
    """Load partial results of general evaluation and accumulate them
        into one dict that will hold the results for the whole dataset
        for provided neural network acrhitecture and power consumption setup.
    
    Args:
        data_root: A string with path to evaluation results.
        arch: A string with model architecture id.
        setup: A string with power consumption id.
    
    Results:
        A tuple of dict with accumulated data, corresponding to
            'arch' and 'setup' args neural network parameters
            following the format described in ml.model.build_model().
    """
    data = {}
    arch_params = ()
    for filename in os.listdir(data_root):
        if not filename.endswith('.pkl'):
            continue
        params = extract_params(filename)
        interested = (arch == get_arch(params) and filename.startswith(setup))
        if not interested:
            continue
        data = accumulate_data(
            os.path.join(data_root, filename),
            data
        )
        arch_params = params
    return data, arch_params

def calc_stats(data):
    """Calculate statistic of mean and standard deviation for spatial accuracy
        results of general evaluation provided in the 'groups' representation.

    Args:
        data: A DataFrame with results in 'groups' representation
            returned from plots.utils.convert_to_groups().

    Returns:
        A dict with aforementioned statistics.
    """
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
    """Accumulate 'groups' representation of partial results of
        general evaluation for provided subject, neural network
        architecture and trainig approach into one provided DataFrame.

    Args:
        groups: A DataFrame to accumulate results in.
        data: A dict that contains results of provided subject.
        subj: A string with subject id.
        arch: A string with model architecture id.
        appr: A string with training approach id. 'ft' and 'scr' are supported.
    
    Returns:
        A DataFrame with accumulated data.
    """
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
    """Convert provided results of general evaluation into
        'groups' representation. Groups are 4 possible pairs of
        neural network architecture ('mlp', 'cnn') and approach
        to train them ('fine-tune', 'from scratch').

    Args:
        mlp_data: A dict with results for 'mlp' architecture
            returned from plot.utils.load_data().
        cnn_data: A dict with results for 'cnn' architecture
            returned from plot.utils.load_data().
    
    Returns:
        A DataFrame with converted data representation.
    """
    groups = pd.DataFrame(columns=['subj', 'arch', 'appr', 'Group', 'rep', 'acc'])

    data = {
        'mlp': mlp_data,
        'cnn': cnn_data
    }
    subjs = mlp_data['subjs']
    for subj in subjs:
        groups = accumulate_group(groups, data, subj, 'mlp', 'ft')
        groups = accumulate_group(groups, data, subj, 'mlp', 'scr')

        groups = accumulate_group(groups, data, subj, 'cnn', 'ft')
        groups = accumulate_group(groups, data, subj, 'cnn', 'scr')

    groups['subj'] = groups['subj'].astype(int)
    groups['arch'] = groups['arch'].astype(int)
    groups['appr'] = groups['appr'].astype(int)
    groups['rep'] = groups['rep'].astype(int)

    return groups
