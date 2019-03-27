import os
import sys

from ml.grid_search import get_best_model_params
from ml.load_data import get_general_data, get_specific_data, get_calib_like_data, split_test_from_all
from ml.model import build_model
from ml.utils import get_model_path, default_config_if_none
from utils.utils import get_arch

def load_and_finetune(root, train_subjs, subj, params, learning_config=None):
    learning_config = default_config_if_none(learning_config)

    arch = get_arch(params)

    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_calib_like_data(root, subj, arch)

    model = build_model(params)
    model_path = get_model_path(train_subjs, params)
    model.load_weights(model_path)

    if arch == 'cnn':
        model.freeze_conv()

    print('Model ' + model_path + ' loaded')

    fit_time = model.train(
        X_train, y_train, X_val, y_val,
        **learning_config
    )

    print('Partial fit completed')
    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

    return train_acc, test_acc, fit_time

def find_train_test_split(subj):
    subjs_split = [
        ['1', '2', '3', '4'],
        ['5', '6', '7', '8'],
        ['9', '10', '11', '12'],
        ['13', '14', '15', '16'],
        ['17', '18', '19', '20'],
        ['21', '22', '23']
    ]

    for split in subjs_split:
        if subj in split:
            return split_test_from_all(split)
    return None

def calib_for_train(root, subj):
    results = {}
    for setup in ['lp', 'hp']:
        params = get_best_model_params('cnn', setup)
        train_subjs, _ = find_train_test_split(subj)
        bg_bs_config = {
            'batch_size': 2000,
            'patience': 10
        }
        _, acc, _ = load_and_finetune(root, train_subjs, subj, params, bg_bs_config)
        results[setup] = acc
    return results

if __name__ == '__main__':
    bad_split = calib_for_train(sys.argv[1], '6')
    good_split = calib_for_train(sys.argv[1], '8')
    print(bad_split)
    print(good_split)