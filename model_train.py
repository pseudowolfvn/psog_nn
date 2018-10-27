import os

import json
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.losses import mean_absolute_error
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.regularizers import l2
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

from utils.utils3 import Clf, calc_etdq
from utils.sensor3 import Sensor


ROOT = r'D:\DmytroKatrychuk\dev\research\psog_dk\dataset'
DESIGN = 'grid15'
TRAIN_EXP = os.path.join(ROOT, 'train')
TEST_EXP = os.path.join(ROOT, 'test')


def get_train_data():
    with open(os.path.join('designs', DESIGN + '.json'), 'r') as f:
        sensor = Sensor(json.load(f))

    X_train = []
    y_train = []
    for exp in os.listdir(TRAIN_EXP):
        if not exp.endswith('_{}.csv'.format(DESIGN)):
            continue
        path = os.path.join(TRAIN_EXP, exp)
        data = pd.read_csv(path, sep='\t', index_col=0)
   
        X = data[sensor.sr_names].values
        y = data[['posx', 'posy']].values

        X_train.extend(X)
        y_train.extend(y)

        print(path, ' added to train set')

    return np.array(X_train), np.array(y_train)


def get_test_data(exp_num=None, filter_data=False):
    with open(os.path.join('designs', DESIGN + '.json'), 'r') as f:
        sensor = Sensor(json.load(f))

    X_test = []
    y_test = []
    for exp in os.listdir(TEST_EXP):
        if not exp.endswith('_{}.csv'.format(DESIGN)):
            continue
        path = os.path.join(TEST_EXP, exp)
        data = pd.read_csv(path, sep='\t', index_col=0)

        if filter_data:
            w = 3
            _f = data[sensor.sr_names].rolling(window=w).mean()
            data.loc[w-1:, sensor.sr_names] = _f.loc[w-1:]

        X = data[sensor.sr_names].values
        y = data[['posx', 'posy']].values

        X_test.extend(X)
        y_test.extend(y)

        print(path, ' added to test set')

    return np.array(X_test), np.array(y_test)


def calc_spatial_scores(model, X, y_gt):
    y_pred = model.predict(X)

    ROOT_UIST = '%s\\UIST\\Hybrid_PSV_Simulations\\MATLAB_Sims'%TEST_EXP
    GT_DATA_UIST = 'Hybrid_PSV_Data\\Groundtruth_Data'
    GT_EXP = 'TX'
    fixations_manual = scio.loadmat('%s\\%s\\manualCleanFixations_%s.mat' % \
                                (ROOT_UIST, GT_DATA_UIST, GT_EXP))['fixations']

    acc, rms = calc_etdq(y_gt, y_pred, fixations_manual)

    err_test = y_gt - y_pred
    mae = np.mean(np.hypot(err_test[:,0], err_test[:,1]))
    return acc, mae


def train_sklearn_model(X, y, arch=None):
    if arch is None:
        arch = [128,16,96]
    model_params = {"random_state": 0o62217, "warm_start": False, "verbose": False,
                  "hidden_layer_sizes": arch, "activation": "relu",
                  "shuffle": True, "batch_size": "auto", "validation_fraction": 0.1,
                  "solver": "adam",
                  "learning_rate_init": 0.001, "max_iter": 500,
                  "learning_rate": "constant", "early_stopping": True,
                  "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-08,
                  "tol": 0.0001, "alpha": 0.0001
                   }
    model = MLPRegressor(**model_params)
    model.fit(X, y)
    return model


def train_keras_model(X, y):
    model = Sequential()
    #model.add(Conv2D(8, 3))
    #model.add(Conv2D(4, 3, padding='same'))
    #model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(96, activation='relu'))
    #model.add(Dropout(0.5))

    #model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(2, kernel_regularizer=l2(0.0001)))

    nadam_opt = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_absolute_error', optimizer=nadam_opt)

    early_stopping = EarlyStopping(monitor='val_loss'
        , min_delta=0.0001, patience=10, mode='auto')

    print('Training...')
    model.fit(X_train, y_train, nb_epoch=500, batch_size=200
        , validation_split=0.1, verbose=2
        , callbacks=[])
    return model

X_train, y_train = get_train_data()
X_test, y_test = get_test_data(filter_data=False)

print(X_train.shape)
print(X_test.shape)

#X_train = X_train.reshape((X_train.shape[0], 3, 5, 1))
#X_test = X_test.reshape((X_test.shape[0], 3, 5, 1))

normalizer = PCA(whiten=True, random_state=0o62217)
normalizer.fit(X_train)
X_train = normalizer.transform(X_train)
X_test = normalizer.transform(X_test)

# model = train_sklearn_model(X_train, y_train)

# acc, mae = calc_spatial_scores(model, X_test, y_test)
# print('ACC: ', acc)
# print('MAE: ', mae)

model = train_sklearn_model(X_train, y_train, arch=[12]*6)
acc, mae = calc_spatial_scores(model, X_test, y_test)
print('Model fited with ACC = {}, MAE = {} .'.format(acc, mae))

exit()

## GRID SEARCH
file = open('full_grid_search_early_stop.log', 'a')
models = []
for layers in range(3, 6 + 1):
    for neurons in range(4, 64 + 1):
        arch = [neurons]*layers
        print('Fitting the model with arch: ' + str(arch), file=file)
        model = train_sklearn_model(X_train, y_train, arch=arch)
        acc, mae = calc_spatial_scores(model, X_test, y_test)
        print('Model fited with ACC = {}, MAE = {} .'.format(acc, mae), file=file)
        models.append((acc, mae, layers, neurons))
        file.flush()

models.sort()
for model in models:
    print(model, file=file)

file.close()

