import os
import sys

import numpy as np
from sklearn.externals import joblib

from ml.finetune import load_and_finetune
from ml.eval import StudyEvaluation
from ml.load_data import get_loo_subjs_split
from ml.utils import get_module_prefix, default_config_if_none

class LOOEvaluation(StudyEvaluation):
    def __init__(self, root):
        super().__init__(root, ['cnn'], ['lp'], study_id='loo_zero')

    def _evaluate_approaches(
        self, train_subjs, test_subjs,
        params, setup, config, REPS, redo
    ):
        config = default_config_if_none(config)

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

        config['epochs'] = 0

        data = {'subjs': test_subjs}
        for subj in test_subjs:
            ft = np.zeros((REPS))
            ft_time = np.zeros((REPS))

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

        print(data)

        joblib.dump(data, data_path)

        return data


    def run(self, learning_config=None, reps=10):
        return super().run(learning_config, reps, get_loo_subjs_split)

if __name__ == '__main__':
    eval = LOOEvaluation(sys.argv[1])
    eval.run(reps=1)
