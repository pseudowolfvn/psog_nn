""" Basic analysis of using calibration-like distribution of training set.
"""
import os
import sys

from ml.finetune import load_and_finetune
from ml.from_scratch import train_from_scratch
from ml.grid_search import get_best_model_params
from ml.load_data import get_calib_like_data, find_train_test_split


def evaluate_subj(root, subj):
    """Evaluate calibration-like data as the training set
    for specific subject.

    Args:
        root: A string with path to dataset.
        subj: A string with specific subject id.
    """
    results = {}
    for setup in ['lp', 'hp']:
        params = get_best_model_params('cnn', setup)
        train_subjs, _ = find_train_test_split(subj)
        bg_bs_config = {
            'batch_size': 2000,
            'patience': 50
        }
        _, acc, _ = load_and_finetune(
            root, train_subjs, subj,
            params, bg_bs_config, get_calib_like_data
        )
        # _, acc, _ = train_from_scratch(
        #     root, subj, params,
        #     bg_bs_config, get_calib_like_data
        # )
        results[setup] = acc
    return results

def evaluate_calib(root, subj_ids):
    """Evaluate calibration-like data as the training set
    for provided list of subjects.

    Args:
        root: A string with path to dataset.
        subj_ids: A list with subjects ids.
    """
    results = {}
    for subj_id in subj_ids:
        results[subj_id] = evaluate_subj(root, subj_id)
    
    print('Accuracy on testing set with using calibration-like training set:')
    for k, v in results.items():
        print('Subject', k, ':', v)

if __name__ == '__main__':
    evaluate_calib(sys.argv[1], ['6', '8'])