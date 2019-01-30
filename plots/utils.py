from itertools import cycle
import os

from sklearn.externals import joblib

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
    data = {**data, **temp}
    return data

def extract_params(name):
    opened = name.find('(')
    closed = name.find(')')
    return tuple(map(int, name[opened + 1: closed].split(', ')))

def load_data(data_root, arch, setup):
    data = {}
    for filename in os.listdir(data_root):
        interested = arch in filename and
            setup in filename and 
            filename.endswith('.pkl')
        if not interested:
            continue
        data = accumulate_data(
            os.path.join(data_root, filename),
            data
        )
        arch_params = extract_params(filename)
    return data, arch_params

def calc_stats(data):
    stats = {
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
    return stats