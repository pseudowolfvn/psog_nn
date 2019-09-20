""" Machine-learning utility functions.
"""
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.externals import joblib


def default_config_if_none(learning_config):
    """Get default training config.

    Args:
        learning_config: A dict with training process parameters. All keyword
            arguments of ml.Model.train() are supported as its keys.

    Returns:
        A dict passed by 'learning_config' arg if it's not None
            otherwise default training config.
    """
    if learning_config is None:
        learning_config = {
            'batch_size': 200,
            'epochs': 1000,
            'patience': 50,
        }
    return learning_config

def get_module_prefix():
    """Get current module prefix.
    
    Returns:
        A string with module prefix.
    """
    return 'ml'

def subjs_list_to_str(subjs):
    subjs_str = ''
    N = len(subjs)
    for i in range(N):
        subjs_str += '_' + subjs[i]
    return subjs_str

def get_model_path(subjs, params, dataset_id=None):
    """Get corresponding model path realtive to the project's root.

    Args:
        subjs: A list of subjects ids the model was trained on.
        params: A tuple with neural network paramters following
            the format described in ml.model.build_model().

    Returns:
        A string with model path.
    """
    model_dir = os.path.join(get_module_prefix(), 'models')
    if dataset_id:
        model_dir = os.path.join(model_dir, dataset_id)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
    return os.path.join(
        model_dir,
        'keras_' + str(params) + subjs_list_to_str(subjs) + '.h5'
    )

def get_normalizer_path(arch, train_subjs):
    return os.path.join(
        get_module_prefix(), 'pca',
        arch + subjs_list_to_str(train_subjs) + '.pkl'
    )

def filter_outliers(data, verbose=False):
    """Remove outliers from the data.

    Args:
        data: A pandas DataFrame following the format of 
            preproc.shift_and_crop_subj() output.
        verbose: A boolean that shows if removed data is printed to console.

    Returns:
        A pandas DataFrame with removed outliers.
    """
    outliers = data[
        (data.pos_x.abs() > 20.)
        | (data.pos_y.abs() > 20)
    ]
    if verbose:
        print(outliers)
    return data.drop(outliers.index)

def normalize(X_train, X_test, arch, train_subjs=None):
    """Do data whitening and additional dimensionality reduction with PCA
        in case of 'mlp' architecture.

    Args:
        X_train: An array of training set values with
            PSOG sensor raw outputs.
        X_test: An array with test set.
        arch: A string with model architecture id.
        train_subjs: A list of subjects ids the network was pre-trained on.

    Returns:
        A tuple of training set, test set modified in aforementioned way.
    """
    # we need to dump and load PCA (or only whitening in case of CNN)
    # results if any pool of train_subjs is provided
    should_load = train_subjs is not None

    if should_load:
        norm_dir = os.path.join(get_module_prefix(), 'pca')
        if not os.path.exists(norm_dir):
            os.mkdir(norm_dir)
        norm_path = get_normalizer_path(arch, train_subjs)

    # we only want to do data whitening for CNN architecture
    pca_params = {
        'n_components': None if arch == 'cnn' else 0.99,
        'whiten': True,
        'random_state': 0o62217
    }

    if should_load and os.path.exists(norm_path):
        normalizer = joblib.load(norm_path)
    else:
        normalizer = PCA(**pca_params)
        normalizer.fit(X_train)
        if should_load:
            joblib.dump(normalizer, norm_path)

    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test

def report_results(data, report_name=None):
    results = np.zeros((len(data['subjs']), 4))
    for i, subj in enumerate(data['subjs']):
        ft_mean = np.mean(data[subj]['ft']['data'])
        ft_std = np.std(data[subj]['ft']['data'])
        scr_mean = np.mean(data[subj]['scr']['data'])
        scr_std = np.std(data[subj]['scr']['data'])
        results[i] = [ft_mean, ft_std, scr_mean, scr_std]
    
    report = pd.DataFrame(
        results,
        columns=['Ft, mean', 'Ft, std', 'Scr, mean', 'Scr, std']
    )

    report_dir = 'tmp'
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    if report_name is None:
        report_name = datetime.datetime.now().isoformat().replace(':', '-')

    report.to_csv(os.path.join(report_dir, report_name + '.csv'), index=False)
