""" Evaluate approaches per architecture per setup.
"""
import os
import sys

import numpy as np
from sklearn.externals import joblib

from ml.finetune import train_and_save, load_and_finetune
from ml.from_scratch import train_from_scratch
from ml.grid_search import get_best_model_params
from ml.load_data import split_test_from_all, default_split_if_none
from ml.utils import get_module_prefix
from utils.utils import get_arch


class StudyEvaluation:
    """Class that provides access to general evaluation of training approaches
    for different neural network architectures and power consumption setups.
    """

    def __init__(self, root, archs, setups, study_id='', redo=True):
        """Inits StudyEvaluation with path to dataset and study configuration.

        Args:
            root: A string with path to dataset.
            archs: A list with neural network architectures to evaluate.
            setups: A list with power consumption setups to evaluate.
            study_id: A string with study id
                (appended to file with study results).
            redo: A boolean that shows if study should be done again
                if files of results already exist.
        """
        self.root = root
        self.archs = archs
        self.setups = setups
        self.study_id = str(study_id)
        self.redo = redo

        self.results = {}
    
    def pretrain_model(self, train_subjs, test_subjs, params):
        """Pre-train the neural network model on provided set of subjects.

        Args:
            train_subjs: A list of subjects ids to train on.
            test_subjs: A list of subjects ids to test on.
            params: A tuple with neural network paramters following the format
                described in ml.model.build_model().
        """
        train_and_save(self.root, train_subjs, test_subjs, params, load=True)

    def _get_study_prefix(self, config):
        """Get prefix of the study depending on its id.

        Args:
            config: A dict that has 'batch_size' field with the
                corresponding value used in this study.

        Returns:
            A string with prefix.
        """
        if self.study_id == '':
            return ''
        else:
            return self.study_id + '_' + str(config['batch_size']) + 'bs_'

    def _evaluate_approaches(
        self, train_subjs, test_subjs,
        params, setup, config, REPS, redo
    ):
        """Evalute spatial accuracies and times spent for training
            for 'fine-tune' and 'from scratch' approaches
            for provided study configuration.

        Args:
            train_subjs: A list of subjects ids to train on.
            test_subjs: A list of subjects ids to test on.
            params: A tuple with neural network paramters following the format
                described in ml.model.build_model().
            setup: A string with power consumption setup to evaluate.
            config: A dict with parameters used for training following
                the format described in ml.utils.default_config_if_none().
            REPS: An int with number of repetitions
                of training for each subject.
            redo: A boolean that shows if study should be done again
                if files of results already exist.

        Returns:
            A dict with results following the format of self.run() return.
        """
        results_dir = os.path.join(get_module_prefix(), 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        name_prefix = self._get_study_prefix(config)
        data_name = name_prefix + str(setup) + '_' + str(params) + '_' + \
            str(test_subjs) + '.pkl'
        data_path = os.path.join(results_dir, data_name)

        if not redo and os.path.exists(data_path):
            print('Evaluation results', data_path, 'already exists, skip')
            data = joblib.load(data_path)
            return data

        data = {'subjs': test_subjs}
        for subj in test_subjs:
            ft = np.zeros((REPS))
            ft_time = np.zeros((REPS))
            scr = np.zeros((REPS))
            scr_time = np.zeros((REPS))

            for i in range(REPS):
                _, acc, t = load_and_finetune(
                    self.root, train_subjs, subj,
                    params, config
                )
                ft[i] = acc
                ft_time[i] = t

            data[subj] = {}
            data[subj]['ft'] = {}
            data[subj]['ft']['data'] = ft
            data[subj]['ft']['time'] = ft_time

            for i in range(REPS):
                _, acc, t = train_from_scratch(
                    self.root, subj, params, config
                )
                scr[i] = acc
                scr_time[i] = t
            data[subj]['scr'] = {}
            data[subj]['scr']['data'] = scr
            data[subj]['scr']['time'] = scr_time

        print(data)

        joblib.dump(data, data_path)

        return data

    def _accumulate_results(self, data):
        """Accumulate several dicts returned from self._evalute_approaches()
            into one self.results dict.

        Args:
            data: A dict with partial results returned
                from self._evalute_approaches().
        """
        for k, v in data.items():
            if k == 'subjs' and 'subjs' in self.results:
                self.results[k].extend(v)
            else:
                self.results[k] = v

    def run(self, learning_config=None, reps=10, split_source=None):
        """Main method to run the general study evaluation
            for the whole dataset.

        Args:
            learning_config: A dict with parameters used for training following
                the format described in ml.utils.default_config_if_none().
            reps: An int with number of repetitions
                of training for each subject.

        Returns:
            A dict with results in the format: {
                'subjs': [<subj_ids>],
                <foreach subj_id from subj_ids>: {
                    'ft': {
                        'data': [<spatial accuracies
                            for 'fine-tune' approach>],
                        'time': [<times spent for training
                            for fine-tune approach>]
                    },
                    'scr': {
                        'data': [<spatial accuracies 
                            for 'from scratch' approach>],
                        'time': [<times spent for training
                            for 'from scratch' approach>]
                    }
                }
            }
        """
        # Workaround to be able to call this method for the same object
        # more than once. Without it, after first call, first element 
        # of the following array containes all IDs.
        # TODO: find the source of bug
        split_source = default_split_if_none(split_source)

        subjs_split = split_source()
        self.results = {}

        for arch in self.archs:
            for setup in self.setups:
                params = get_best_model_params(arch, setup)
                for test_subjs in subjs_split:
                    print(
                        'Evaluate approaches for params: ', params,
                        ', setup:', setup
                    )
                    train_subjs, test_subjs = split_test_from_all(test_subjs)

                    self.pretrain_model(train_subjs, test_subjs, params)

                    data = self._evaluate_approaches(
                        train_subjs, test_subjs,
                        params, setup, learning_config, reps, self.redo 
                    )
                    self._accumulate_results(data)

        return self.results

