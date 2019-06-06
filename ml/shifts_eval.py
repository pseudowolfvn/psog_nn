import os
import sys

import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.externals import joblib

from ml.finetune import load_and_finetune
from ml.eval import StudyEvaluation
from ml.load_data import get_shifts_split_data, get_data, \
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

def shifts_randn_5_eval(root):
    eval = StudyEvaluation(
        root, ['cnn'], ['lp'],
        study_id='shifts_randn_5',
        redo=False
    )
    return eval.run(
        learning_config={'batch_size': 2000}, reps=5,
        split_source=get_loo_subjs_split
    )

def shifts_outer_eval(root, rad_gt, rad_lt):
    eval = StudyEvaluation(
        root, ['cnn'], ['lp'],
        study_id='shifts_outer_' + str(rad_gt) + '_' + str(rad_lt),
        redo=False
    )
    return eval.run(
        learning_config={'batch_size': 2000}, reps=5,
        split_source=get_loo_subjs_split,
        data_source=make_get_shifts_outer_split_data(rad_gt, rad_lt)
    )

def shifts_outer_cumulative(root):
    report_results(shifts_outer_eval(root, 0.0, 1.0), 'shifts_0.0_1.0')
    report_results(shifts_outer_eval(root, 1.0, 1.5), 'shifts_1.0_1.5')
    report_results(shifts_outer_eval(root, 1.0, 2.0), 'shifts_1.0_2.0')
    report_results(shifts_outer_eval(root, 1.0, np.inf), 'shifts_1.0_inf')

def shifts_outer_exclusive(root):
    report_results(shifts_outer_eval(root, 0.0, 1.0), 'shifts_0.0_1.0')
    report_results(shifts_outer_eval(root, 1.0, 1.5), 'shifts_1.0_1.5')
    report_results(shifts_outer_eval(root, 1.5, 2.0), 'shifts_1.5_2.0')
    report_results(shifts_outer_eval(root, 2.0, np.inf), 'shifts_2.0_inf')


def visualize(root):
    X_train, y_train = get_data(root, with_shifts=True)
    dist = lambda x, y: np.sqrt(x**2 + y**2)
    test_ind = np.where([
        dist(x, y) > 2. and dist(x, y) <= np.inf
            for x, y in X_train[:, -2:]
    ])[0]
    plot([
        go.Box(y=X_train[:, -2][test_ind], name='Hor'),
        go.Box(y=X_train[:, -1][test_ind], name='Ver')
    ], filename='shifts_inf_boxplot')
    
    # fig, ax = plt.subplots()
    # ax.scatter(X_train[:, -2][train_ind], X_train[:, -1][train_ind], marker='x', c='r')
    # ax.scatter(X_train[:, -2][test_ind], X_train[:, -1][test_ind], marker='x', c='b', s=0.5)
    # ax.scatter(y_train[:, 0][train_ind], y_train[:, 1][train_ind], marker='x', c='r', s=0.5)
    # ax.scatter(y_train[:, 0][test_ind], y_train[:, 1][test_ind], marker='x', c='b', s=0.5)
    # plt.show()

if __name__ == '__main__':
    root = sys.argv[1]
    shifts_outer_exclusive(root)
    # report_results(shifts_randn_5_eval(root))

    # visualize(root)
