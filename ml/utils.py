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

def normalize(X_train, X_test, arch, train_subjs=None):
    # we need to dump and load PCA results 
    # if any pool of train_subjs is provided
    # and architeture is not CNN
    should_load = train_subjs is not None and arch != 'cnn'

    if should_load:
        norm_dir = os.path.join(get_module_prefix(), 'pca')
        if not os.path.exists(norm_dir):
            os.mkdir(norm_dir)
        norm_path = os.path.join(
            norm_dir,
            'normalizer_' + str(train_subjs) + '.pkl'
        )

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
