""" Evaluate approaches per architecture per setup.
"""
import os
import sys

import numpy as np
from sklearn.externals import joblib

from ml.finetune import train_and_save, load_and_finetune, load_and_finetune_fc
from ml.from_scratch import train_from_scratch
from ml.grid_search import get_best_model_params
from ml.load_data import split_test_from_all
from ml.utils import get_module_prefix
from utils.utils import get_arch


def evaluate_approaches(root, test_subjs, params, setup, redo, REPS=10):
    results_dir = os.path.join(get_module_prefix(), 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    data_path = os.path.join(
        results_dir,
        str(setup) + '_' + str(params) + '_' + str(test_subjs) + '.pkl'
    )

    if not redo and os.path.exists(data_path):
        print('Evaluation results', data_path, 'already exists, skip')
        return

    data = {'subjs': test_subjs}
    for subj in test_subjs:
        ft = np.zeros((REPS))
        ft_time = np.zeros((REPS))
        ft_fc = np.zeros((REPS))
        ft_fc_time = np.zeros((REPS))
        scr = np.zeros((REPS))
        scr_time = np.zeros((REPS))

        for i in range(REPS):
            _, acc, t = load_and_finetune(root, test_subjs, subj, params)
            ft[i] = acc
            ft_time[i] = t

        data[subj] = {}
        data[subj]['ft'] = {}
        data[subj]['ft']['data'] = ft
        data[subj]['ft']['time'] = ft_time

        if get_arch(params) == 'cnn':
            for i in range(REPS):
                _, acc, t = load_and_finetune_fc(root, test_subjs, subj, params)
                ft_fc[i] = acc
                ft_fc_time[i] = t
            data[subj]['ft_fc'] = {}
            data[subj]['ft_fc']['data'] = ft_fc
            data[subj]['ft_fc']['time'] = ft_fc_time

        for i in range(REPS):
            _, acc, t = train_from_scratch(root, subj, params)
            scr[i] = acc
            scr_time[i] = t
        data[subj]['scr'] = {}
        data[subj]['scr']['data'] = scr
        data[subj]['scr']['time'] = scr_time

    print(data)

    joblib.dump(data, data_path)

def cross_testing(root, test_subjs, params, setup, redo):
    print(
        'Evaluate approaches for params: ', params,
        ', setup:', setup
    )

    train_subjs, test_subjs = split_test_from_all(test_subjs)
    print('Train on: ', train_subjs, 'Test on: ', test_subjs)

    train_and_save(root, train_subjs, test_subjs, params, load=True)
    evaluate_approaches(root, test_subjs, params, setup, redo)

def evaluate_study(root, archs, setups, redo=True):
    subjs_split = [
        ['1', '2', '3', '4'],
        ['5', '6', '7', '8'],
        ['9', '10', '11', '12'],
        ['13', '14', '15', '16'],
        ['17', '18', '19', '20'],
        ['21', '22', '23']
    ]
    for arch in archs:
        for setup in setups:
            params = get_best_model_params(arch, setup)
            for subjs in subjs_split:
                cross_testing(root, subjs, params, setup, redo)

if __name__ == "__main__":
    #evaluation(['mlp', 'cnn'], ['lp', 'hp'])
    evaluate_study(sys.argv[1], ['mlp'], ['lp'])
