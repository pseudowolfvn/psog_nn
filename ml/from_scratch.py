""" 'From scratch' approach related training.
"""
from ml.load_data import get_specific_data
from ml.model import build_model
from ml.utils import get_model_path
from utils.utils import get_arch


def train_from_scratch(root, subj, params):
    # TODO: change True back to False
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_specific_data(root, subj, get_arch(params))

    model = build_model(params)
    fit_time = model.train(
        X_train, y_train, X_val, y_val,
        batch_size=2000
    )

    print('Model ' + get_model_path(subj, params) + ' trained from scratch')
    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

    return train_acc, test_acc, fit_time
