""" Basic training time complexity analysis.
"""
import sys

import numpy as np

from ml.eval import StudyEvaluation
from plots.utils import accumulate_data

def calc_time_stats(data):
    def time_arr(data, appr):
        return [data[subj][appr]['time'] for subj in data['subjs']]

    ft_time = time_arr(data, 'ft')
    scr_time = time_arr(data, 'scr')
    time_stats = {
        'ft': {
            'mean': np.mean(ft_time),
            'std': np.std(ft_time),
        },
        'scr': {
            'mean': np.mean(scr_time),
            'std': np.std(scr_time),
        }
    }
    return time_stats

def evaluate_config(eval, config):
    data = eval.run(config, 1)
    stats = calc_time_stats(data)
    print('Time statistics for config: ', config)
    print(stats)
    print()

def evaluate(root, archs, setups, redo=True):
    eval = StudyEvaluation(root, archs, setups, 'time', redo=redo)
    # smaller batch size
    sm_bs_config = {
        'batch_size': 200,
        'patience': 100
    }
    evaluate_config(eval, sm_bs_config)
    # bigger batch size
    bg_bs_config = {
        'batch_size': 2000,
        'patience': 100
    }
    evaluate_config(eval, bg_bs_config)


if __name__ == "__main__":
    evaluate(sys.argv[1], ['cnn'], ['lp'], redo=False)