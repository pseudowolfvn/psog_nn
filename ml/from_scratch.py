""" 'From scratch' approach related training.
"""
from ml.load_data import get_specific_data, get_calib_like_data, \
    default_source_if_none
from ml.model import build_model
from ml.utils import get_model_path, default_config_if_none
from utils.utils import get_arch


def train_from_scratch(root, subj, params,
        learning_config=None, data_source=None):
    """Train the neural network model 'from scratch'
        for provided specific subject.

    Args:
        root: A string with path to dataset.
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
    learning_config = default_config_if_none(learning_config)
    data_source = default_source_if_none(data_source)

    X_train, X_val, X_test, y_train, y_val, y_test = \
        data_source(root, subj, get_arch(params))

    in_dim = None if len(X_train.shape) > 2 else X_train.shape[-1]
    print('DEBUG: ', X_train.shape)
    model = build_model(params, in_dim)
    fit_time = model.fit(
        X_train, y_train, X_val, y_val,
        **learning_config
    )

    print('Model ' + get_model_path(subj, params) + ' trained from scratch')
    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

    return train_acc, test_acc, fit_time
