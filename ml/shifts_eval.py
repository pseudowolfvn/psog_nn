import os
import sys

import numpy as np
from sklearn.externals import joblib

from ml.finetune import load_and_finetune
from ml.eval import StudyEvaluation
from ml.load_data import get_shifts_split_data, \
    make_get_shifts_outer_split_data, get_loo_subjs_split
from ml.utils import get_module_prefix, default_config_if_none, report_results

def shifts_fine_eval(root):
    eval = StudyEvaluation(
        root, ['cnn'], ['lp'],
        study_id='shifts_fine',
        redo=False)
    return eval.run(
        learning_config={'batch_size': 2000}, reps=5,
        data_source=get_shifts_split_data
    )

def shifts_randn_eval(root):
    eval = StudyEvaluation(
        root, ['cnn'], ['lp'],
        study_id='shifts_randn',
        redo=False
    )
    return eval.run(
        learning_config={'batch_size': 2000}, reps=5
    )

def shifts_outer_eval(root, rad):
    eval = StudyEvaluation(
        root, ['cnn'], ['lp'],
        study_id='shifts_outer' + str(rad),
        redo=False
    )
    return eval.run(
        learning_config={'batch_size': 2000}, reps=5,
        split_source=get_loo_subjs_split,
        data_source=make_get_shifts_outer_split_data(rad)
    )

if __name__ == '__main__':
    shifts_outer_eval(sys.argv[1], 1.0)
    shifts_outer_eval(sys.argv[1], 1.5)
    shifts_outer_eval(sys.argv[1], 2.0)
    shifts_outer_eval(sys.argv[1], np.inf)
