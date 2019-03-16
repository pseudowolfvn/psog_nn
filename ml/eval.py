""" Evaluate approaches per architecture per setup.
"""
import os
import sys

import numpy as np
from sklearn.externals import joblib

from ml.finetune import train_and_save, load_and_finetune
from ml.from_scratch import train_from_scratch
from ml.grid_search import get_best_model_params
from ml.load_data import split_test_from_all
from ml.utils import get_module_prefix
from utils.utils import get_arch


class StudyEvaluation:
    def __init__(self, root, archs, setups, study_id='', redo=True):
        self.root = root
        self.archs = archs
        self.setups = setups
        self.study_id = str(study_id)
        self.redo = redo

        self.subjs_split = [
            ['1', '2', '3', '4'],
            ['5', '6', '7', '8'],
            ['9', '10', '11', '12'],
            ['13', '14', '15', '16'],
            ['17', '18', '19', '20'],
            ['21', '22', '23']
        ]
    
    def pretrain_model(self, train_subjs, test_subjs, params):
        train_and_save(self.root, train_subjs, test_subjs, params, load=True)

    def _evaluate_approaches(
        self, train_subjs, test_subjs,
        params, setup, learning_config, REPS, redo
    ):
        results_dir = os.path.join(get_module_prefix(), 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        name_prefix = self.study_id + '_' if len(self.study_id) > 0 else ''
        data_name = name_prefix + str(setup) + '_' + str(params) + '_' + \
            str(test_subjs) + '.pkl'
        data_path = os.path.join(results_dir, data_name)

        if not redo and os.path.exists(data_path):
            print('Evaluation results', data_path, 'already exists, skip')
            return

        data = {'subjs': test_subjs}
        for subj in test_subjs:
            ft = np.zeros((REPS))
            ft_time = np.zeros((REPS))
            scr = np.zeros((REPS))
            scr_time = np.zeros((REPS))

            for i in range(REPS):
                _, acc, t = load_and_finetune(
                    self.root, train_subjs, subj,
                    params, learning_config
                )
                ft[i] = acc
                ft_time[i] = t

            data[subj] = {}
            data[subj]['ft'] = {}
            data[subj]['ft']['data'] = ft
            data[subj]['ft']['time'] = ft_time

            for i in range(REPS):
                _, acc, t = train_from_scratch(
                    self.root, subj, params, learning_config
                )
                scr[i] = acc
                scr_time[i] = t
            data[subj]['scr'] = {}
            data[subj]['scr']['data'] = scr
            data[subj]['scr']['time'] = scr_time

        print(data)

        joblib.dump(data, data_path)

        return data

    def run(self, learning_config=None, reps=10):
        for arch in self.archs:
            for setup in self.setups:
                params = get_best_model_params(arch, setup)
                for test_subjs in self.subjs_split:
                    print(
                        'Evaluate approaches for params: ', params,
                        ', setup:', setup
                    )
                    train_subjs, test_subjs = split_test_from_all(test_subjs)

                    self.pretrain_model(train_subjs, test_subjs, params)

                    self._evaluate_approaches(
                        train_subjs, test_subjs,
                        params, setup, learning_config, reps, self.redo 
                    )

