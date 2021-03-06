""" 'Fine-tune' approach related training.
"""
import os

from ml.load_data import get_general_data, get_specific_data, \
    get_calib_like_data, default_source_if_none
from ml.model import build_model
from ml.utils import get_model_path, default_config_if_none
from utils.utils import get_arch


def train_and_save(root, train_subjs, test_subjs, params, load=False,
        learning_config=None, data_source=None):
    """Pre-train and save the neural network model on provided set of subjects.

    Args:
        root: A string with path to dataset.
        train_subjs: A list of subjects ids to train on.
        test_subjs: A list of subjects ids to test on.
        params: A tuple with neural network paramters following
            the format described in ml.model.build_model().
        load: A boolean that shows if network should be trained again
            if files with weights already exist.
        learning_config: A dict with parameters used for training following
            the format described in ml.utils.default_config_if_none().
        data_source: A function object that is used to load data.
            ml.load_data.get_specific_data and ml.load_data.get_calib_like_data
            are supported.
    """
    learning_config = default_config_if_none(learning_config)
    data_source = default_source_if_none(data_source)

    model_path = get_model_path(train_subjs, params)

    if load and os.path.exists(model_path):
        print('Model', model_path, 'already exists, skip')
        return

    X_train, X_val, X_test, y_train, y_val, y_test = \
        data_source(root, train_subjs, test_subjs, get_arch(params))

    model.train(X_train, y_train, X_val, y_val, batch_size=2000)
    dim = None if len(X_train.shape) > 2 else X_train.shape[-1]
    learning_config['batch_size'] = 2000
    model = build_model(params, in_dim=dim, learning_config=learning_config)

    model.fit(X_train, y_train, X_val, y_val)
    model.save_weights(model_path)
    print('Model', model_path, ' saved')

    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

def load_and_finetune(root, train_subjs, subj, params,
        learning_config=None, data_source=None):
    """Load pre-trained model and 'fine-tune' it for provided
        specific subject.

    Args:
        root: A string with path to dataset.
        train_subjs: A list of subjects ids the network was pre-trained on.
        subj: A string with specific subject id.
        params: A tuple with neural network paramters following
            the format described in ml.model.build_model().
        learning_config: A dict with parameters used for training following
            the format described in ml.utils.default_config_if_none().
        data_source: A function object that is used to load data.
            ml.load_data.get_specific_data and ml.load_data.get_calib_like_data
            are supported.

    Returns:
        A tuple with spatial accuracies on train set, test set
            and time spent for training.
    """
    config = default_config_if_none(learning_config)
    data_source = default_source_if_none(data_source)

    arch = get_arch(params)

    X_train, X_val, X_test, y_train, y_val, y_test = \
        data_source(root, subj, arch, train_subjs)

    dim = None if len(X_train.shape) > 2 else X_train.shape[-1]
    model = build_model(params, in_dim=dim, learning_config=config)
    model_path = get_model_path(train_subjs, params)
    model.load_weights(model_path)

    if arch == 'cnn':
        model.freeze_conv()

    print('Model ' + model_path + ' loaded')

    fit_time = model.fit(X_train, y_train, X_val, y_val)

    print('Partial fit completed')
    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

    return train_acc, test_acc, fit_time
