import os

from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def get_module_prefix():
    return r'.\ml'

def get_model_name(subjs, params):
    return os.path.join(
        MODULE_PREFIX,
        'models',
        'keras_' + str(params) + '_' + str(subjs) + '.h5'
    )

def filter_outliers(data, verbose=True):
    outliers = data[(data.posx.abs() > 20.) 
        | (data.posy.abs() > 20)]
    if verbose:
        print(outliers)
    return data.drop(outliers.index)

def normalize(X_train, X_test, subjs, arch, load=True):
    norm_path = os.path.join(
        get_module_prefix(),
        'pca',
        'normalizer_' + str(subjs) + '.pkl')
    if not load or not os.path.exists(norm_path):
        normalizer = PCA(
            n_components=(None if arch == 'cnn' else 0.99),
            whiten=True,
            random_state=0o62217)
        normalizer.fit(X_train)
        joblib.dump(normalizer, norm_path)
    else:
        normalizer = joblib.load(norm_path)
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test