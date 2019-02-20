""" 'Fine-tune' approach related training.
"""
import os

from ml.load_data import get_general_data, get_specific_data
from ml.model import build_model
from ml.utils import get_model_path
from utils.utils import get_arch


def train_and_save(root, train_subjs, test_subjs, params, load=False):
    model_path = get_model_path(test_subjs, params)
    model = build_model(params)

    if load and os.path.exists(model_path):
        print('Model', model_path, 'already exists, skip')
        return

    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_general_data(root, train_subjs, test_subjs, get_arch(params))

    model.train(X_train, y_train, X_val, y_val, batch_size=2000)
    model.save_weights(model_path)
    print('Model', model_path, ' saved')

    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

def load_and_finetune(root, test_subjs, subj, params):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_specific_data(root, subj, get_arch(params))

    model = build_model(params)
    model_path = get_model_path(test_subjs, params)
    model.load_weights(model_path)

    print('Model ' + model_path + ' loaded')

    fit_time = model.train(
        X_train, y_train, X_val, y_val,
        epochs=1000,
        batch_size=200,
        patience=50
    )

    print('Partial fit completed')
    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

    return train_acc, test_acc, fit_time

def load_and_finetune_fc(root, test_subjs, subj, params):
    if get_arch(params) != 'cnn':
        print('Can be called only for CNN!')
        return

    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_specific_data(root, subj, 'cnn')

    model = build_model(params)
    model_path = get_model_path(test_subjs, params)
    model.load_weights(model_path)

    model.freeze_conv()

    print('Model ' + model_path + ' loaded')

    fit_time = model.train(
        X_train, y_train, X_val, y_val,
        epochs=1000,
        batch_size=2000,
        patience=50
    )

    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

    return train_acc, test_acc, fit_time
