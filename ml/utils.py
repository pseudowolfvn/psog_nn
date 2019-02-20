""" Machine-learning utility functions.
"""
import os

from sklearn.decomposition import PCA
from sklearn.externals import joblib


def get_module_prefix():
    return 'ml'

def get_model_path(subjs, params):
    return os.path.join(
        get_module_prefix(),
        'models',
        'keras_' + str(params) + '_' + str(subjs) + '.h5'
    )

def filter_outliers(data, verbose=False):
    outliers = data[
        (data.pos_x.abs() > 20.)
        | (data.pos_y.abs() > 20)
    ]
    if verbose:
        print(outliers)
    return data.drop(outliers.index)

def normalize(X_train, X_test, arch):
    # we only want to do data whitening for CNN architecture
    pca_params = {
        'n_components': None if arch == 'cnn' else 0.99,
        'whiten': True,
        'random_state': 0o62217
    }
    
    normalizer = PCA(**pca_params)
    normalizer.fit(X_train)

    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test
