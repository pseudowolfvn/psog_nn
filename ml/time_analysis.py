""" Basic training time complexity analysis.
"""
import sys

import numpy as np

from ml.eval import StudyEvaluation
from plots.utils import accumulate_data


def calc_stats(data, field):
    """Calculate statistic of mean and standard deviation for provided
        quantity of results.

    Args:
        data: A dict with results returned from StudyEvaluation.run().
        field: A string with measured quantity.
            'data' for spatial accuracy and
            'time' for time spent for training are supported.

    Returns:
        A dict with aforementioned statistics.
    """
    def arr(data, appr):
        return [data[subj][appr][field] for subj in data['subjs']]

    ft = arr(data, 'ft')
    scr = arr(data, 'scr')
    stats = {
        'ft': {
            'mean': np.mean(ft),
            'std': np.std(ft),
        },
        'scr': {
            'mean': np.mean(scr),
            'std': np.std(scr),
        }
    }
    return stats

def evaluate_config(eval, config):
    """Run the training time evaluation for provided training configuration.

    Args:
        eval: A properly initialized instance of ml.eval.StudyEvaluation class.
        config: A dict with training configuration following the format of
            StudyEvaluation.run().
    """
    data = eval.run(config, 1)
    time_stats = calc_stats(data, 'time')
    acc_stats = calc_stats(data, 'data')
    print('Time statistics for config: ', config)
    print(time_stats)
    print('Accuracy statistics for config: ', config)
    print(acc_stats)
    print()

def evaluate_time(root, archs, setups, redo=True):
    """Run the training time evaluation for the whole dataset.

    Args:
        root: A string with path to dataset.
        archs: A list with neural network architectures to evaluate.
        setups: A list with power consumption setups to evaluate.
        redo: A boolean that shows if evaluation should be done again
            if files of results already exist.
    """
    eval = StudyEvaluation(root, archs, setups, 'time', redo=redo)
    # smaller batch size
    sm_bs_config = {
        'batch_size': 200,
        'patience': 50,
        'epochs': 2000,
    }
    evaluate_config(eval, sm_bs_config)
    # bigger batch size
    bg_bs_config = {
        'batch_size': 2000,
        'patience': 50
    }
    evaluate_config(eval, bg_bs_config)

if __name__ == "__main__":
    evaluate_time(sys.argv[1], ['cnn'], ['lp'], redo=False)
