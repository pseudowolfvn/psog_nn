from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import json
import os
import time
import xml.etree.ElementTree as ET

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.losses import mean_squared_error
from keras.models import Sequential, load_model
from keras.optimizers import Nadam
from keras.regularizers import l2
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from utils.sensor3 import Sensor

MODULE_PREFIX = r'.\models_eval\keras_cnn'

ROOT = r'D:\DmytroKatrychuk\dev\research\psog_nn\eet_data_converter\blender_data'
DESIGN = 'grid15'
#TRAIN_EXP = ['1', '2', '4', '5', '6', '7', '8', '10', '12', '13', '14', '16', '17', '18', '20', '21', '22']
#TEST_EXP = ['3', '11', '15', '19', '23']


def filter_outliers(data, verbose=True):
    if verbose:
        print(data[(data.posx.abs() > 20.) 
        | (data.posy.abs() > 20)])
    return data.drop(data[(data.posx.abs() > 20.) 
        | (data.posy.abs() > 20)].index)

def get_train_data(train_exp):
    with open(os.path.join('designs', DESIGN + '.json'), 'r') as f:
        sensor = Sensor(json.load(f))

    X_train = []
    y_train = []
    for exp in train_exp:
        path = os.path.join(ROOT, exp, exp + '_%s.csv'%DESIGN)
        data = pd.read_csv(path, sep='\t', index_col=0)
   
        data = filter_outliers(data)

        X = data[sensor.sr_names].values
        y = data[['posx', 'posy']].values

        X_train.extend(X)
        y_train.extend(y)

    return np.array(X_train), np.array(y_train)


def get_test_data(test_exp, filter_data=False):
    if not isinstance(test_exp, list):
        test_exp = [test_exp]

    with open(os.path.join('designs', DESIGN + '.json'), 'r') as f:
        sensor = Sensor(json.load(f))

    X_test = []
    y_test = []
    for exp in test_exp:
        path = os.path.join(ROOT, exp, exp + '_%s.csv'%DESIGN)
        data = pd.read_csv(path, sep='\t', index_col=0)

        data = filter_outliers(data)

        if filter_data:
            w = 3
            _f = data[sensor.sr_names].rolling(window=w).mean()
            data.loc[w-1:, sensor.sr_names] = _f.loc[w-1:]

        X = data[sensor.sr_names].values
        y = data[['posx', 'posy']].values

        X_test.extend(X)
        y_test.extend(y)

    return np.array(X_test), np.array(y_test)

def get_general_data(train_subjs, test_subjs, mode):
    X_train, y_train = get_train_data(train_subjs)
    X_test, y_test = get_test_data(test_subjs)
    # TODO: change True back to False
    X_train, X_test = normalize(X_train, X_test, test_subjs, True, mode)

    # train_val_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    if mode == 'cnn':
        X_train = X_train.reshape((X_train.shape[0], 3, 5, 1))
        X_val = X_val.reshape((X_val.shape[0], 3, 5, 1))
        X_test = X_test.reshape((X_test.shape[0], 3, 5, 1))
    
    #print(X_train.shape)
    #print(X_val.shape)
    #print(X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_specific_data(test_subjs, subj, load_normalizer, mode):
    X_train, y_train = get_test_data(subj)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.7, random_state=42)
    X_train, X_test = normalize(X_train, X_test, test_subjs, load_normalizer, mode)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    if mode == 'cnn':
        X_train = X_train.reshape((X_train.shape[0], 3, 5, 1))
        X_val = X_val.reshape((X_val.shape[0], 3, 5, 1))
        X_test = X_test.reshape((X_test.shape[0], 3, 5, 1))

    #print(X_train.shape)
    #print(X_val.shape)
    #print(X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize(X_train, X_test, subjs, load, mode):
    norm_path = os.path.join(MODULE_PREFIX, 'normalizer_' + str(subjs) + '.pkl')
    if not load:
        # save spatial information
        components = None if mode == 'cnn' else 0.99
        normalizer = PCA(n_components=components, whiten=True, random_state=0o62217)
        normalizer.fit(X_train)
        joblib.dump(normalizer, norm_path)
    else:
        normalizer = joblib.load(norm_path)
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test

def get_model_name(subjs, params):
    return os.path.join(
        MODULE_PREFIX,
        'keras_general_' + str(params) + '_' + str(subjs) + '.h5'
    )

def get_mode(params):
    return 'mlp' if params[0] == 0 else 'cnn'

def train_keras_model(X, y, conv_layers, conv_kernel, layers, neurons, X_val=None, y_val=None, batch_size=200):
    model = Sequential()

    for _ in range(conv_layers):
        model.add(Conv2D(conv_kernel, 3, padding='same'))

    if conv_layers > 0:
        model.add(Flatten())

    for _ in range(layers):
        model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(0.0001)))

    model.add(Dense(2))

    nadam_opt = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=nadam_opt)

    early_stopping = EarlyStopping(monitor='val_loss'
        , patience=100, mode='auto', restore_best_weights=True)

    print('Training...')
    if X_val is None or y_val is None:
        model.fit(X, y, nb_epoch=1000, batch_size=batch_size
            , validation_split=0.15, verbose=1
            , callbacks=[early_stopping])
    else:
        model.fit(X, y, nb_epoch=1000, batch_size=batch_size
            , validation_data=(X_val, y_val), verbose=1
            , callbacks=[early_stopping])
    return model

def calc_acc(y_gt, y_pred):
    err_test = y_gt - y_pred
    return np.mean(np.hypot(err_test[:,0], err_test[:,1]))

def keras_train_and_save(train_subjs, test_subjs, params, load=False):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_general_data(train_subjs, test_subjs, mode=get_mode(params))

    model_name = get_model_name(test_subjs, params)
    if load and os.path.exists(model_name):
        print('Model ' + model_name + ' already exists')
        model = load_model(model_name)
        model.summary()
        print('Model ' + model_name + ' loaded')
    else:
        model = train_keras_model(X_train, y_train, *params, X_val, y_val, 2000)
        model.summary()
        model.save(model_name)
        print('Model ' + model_name + ' saved')
    
    print('Train acc: ', calc_acc(y_train, model.predict(X_train)))
    print('Test acc: ', calc_acc(y_test, model.predict(X_test)))

def keras_load_and_finetune(test_subjs, subj, params):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_specific_data(test_subjs, subj, True, mode=get_mode(params))

    model = load_model(get_model_name(test_subjs, params))
    model.summary()
    print('Model loaded')
    early_stopping = EarlyStopping(monitor='val_loss'
        , patience=50, mode='auto', restore_best_weights=True)

    fit_time = time.time()
    model.fit(X_train, y_train, nb_epoch=1000, batch_size=200
            , validation_data=(X_val, y_val), verbose=1
            , callbacks=[early_stopping])
    fit_time = time.time() - fit_time

    print('Partial fit completed')
    train_acc = calc_acc(y_train, model.predict(X_train))
    print('Train acc: ', train_acc)
    test_acc = calc_acc(y_test, model.predict(X_test))
    print('Test acc: ', test_acc)
    return train_acc, test_acc, fit_time

def keras_load_and_finetune_fc(test_subjs, subj, params):
    if get_mode(params) != 'cnn':
        print('Can be called only for CNN!')
        return

    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_specific_data(test_subjs, subj, True, mode='cnn')
    
    model = load_model(get_model_name(test_subjs, params))
    model.summary()
    for layer in model.layers:
        if layer.name.startswith('conv2d'):
            layer.trainable = False
    print('Model loaded')
    nadam_opt = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=nadam_opt)
    early_stopping = EarlyStopping(monitor='val_loss'
        , patience=50, mode='auto', restore_best_weights=True)

    fit_time = time.time()
    model.fit(X_train, y_train, nb_epoch=1000, batch_size=2000
            , validation_data=(X_val, y_val), verbose=1
            , callbacks=[early_stopping])
    fit_time = time.time() - fit_time

    train_acc = calc_acc(y_train, model.predict(X_train))
    print('Train acc: ', train_acc)
    test_acc = calc_acc(y_test, model.predict(X_test))
    print('Test acc: ', test_acc)
    return train_acc, test_acc, fit_time


def keras_train_from_scratch(subj, params):
    # TODO: change True back to False
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_specific_data(subj, subj, True, mode=get_mode(params))

    fit_time = time.time()
    model = train_keras_model(X_train, y_train, *params, X_val, y_val, 2000)
    fit_time = time.time() - fit_time
    
    print('Keras model trained')
    train_acc = calc_acc(y_train, model.predict(X_train))
    print('Train acc: ', train_acc)
    test_acc = calc_acc(y_test, model.predict(X_test))
    print('Test acc: ', test_acc)
    return train_acc, test_acc, fit_time


def test_finetuning(test_subjs, params):
    mode = get_mode(params)
    REPS = 1
    data = {'subjs': test_subjs}
    for subj in test_subjs:
        ft = np.zeros((REPS))
        ft_time = np.zeros((REPS))
        ft_fc = np.zeros((REPS))
        ft_fc_time = np.zeros((REPS))
        scr = np.zeros((REPS))
        scr_time = np.zeros((REPS))
        
        # for i in range(REPS):
        #     _, acc, t = keras_load_and_finetune(test_subjs, subj, params)
        #     ft[i] = acc
        #     ft_time[i] = t
        data[subj] = {}
        data[subj]['ft'] = {}
        data[subj]['ft']['data'] = ft
        data[subj]['ft']['time'] = ft_time

        if mode == 'cnn':
            for i in range(REPS):
                _, acc, t = keras_load_and_finetune_fc(test_subjs, subj, params)
                ft_fc[i] = acc
                ft_fc_time[i] = t
            data[subj]['ft_fc'] = {}
            data[subj]['ft_fc']['data'] = ft_fc
            data[subj]['ft_fc']['time'] = ft_fc_time

        for i in range(REPS):
            _, acc, t = keras_train_from_scratch(subj, params)
            scr[i] = acc
            scr_time[i] = t
        data[subj]['scr'] = {}
        data[subj]['scr']['data'] = scr
        data[subj]['scr']['time'] = scr_time

    print(data)
    joblib.dump(data, os.path.join(
            MODULE_PREFIX,
            'time_' + str(mode) + '_' + str(params) + '_' + str(test_subjs) + '.pkl'
        )
    )



def calc_keras_parameters_num(c_l, k, l, n):
    return c_l*k*k + c_l*k*k*n + n*n*(l - 1) + 2*n

def keras_cnn_grid_search(train_subjs, test_subjs):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_general_data(train_subjs, test_subjs, 'cnn')

    file = open(os.path.join(MODULE_PREFIX, 'keras_cnn_grid_search'), 'w')
    models = []
    for conv_layers in [1,2,4]:
        for conv_kernel in [4,8,16]:
            for layers in [3,4,5]:
                for neurons in [16,32,48,64,96,128]:
                    if calc_keras_parameters_num(conv_layers, conv_kernel
                        , layers, neurons) > 32768:
                        continue
                    print('Fitting the model with arch: ' + str([conv_kernel]*conv_layers)
                        + ', ' + str([neurons]*layers), file=file)
                    fit_time = time.time()
                    model = train_keras_model(X_train, y_train
                        , conv_layers, conv_kernel, layers, neurons, X_val, y_val, 2000)
                    fit_time = time.time() - fit_time
                    val_acc = calc_acc(y_val, model.predict(X_val))
                    train_acc = calc_acc(y_train, model.predict(X_train))
                    test_acc = calc_acc(y_test, model.predict(X_test))
                    print('Model fited with val ACC = {}, train ACC = {}, test ACC = {}, time = {}'
                        .format(val_acc, train_acc, test_acc, fit_time),  file=file)
                    models.append((val_acc, train_acc, test_acc, time, conv_layers, conv_kernel, layers, neurons))
                    file.flush()

    models.sort()
    for model in models:
        print(model, file=file)

    file.close()

def split_test_from_train(test_exp):
    data = [str(i) for i in range(1, 23 + 1) if i != 9]
    train_exp = []
    for exp in data:
        if exp not in test_exp:
            train_exp.append(exp)
    return train_exp, test_exp

def cross_testing(test_subjs, params):
    train_subjs, test_subjs = split_test_from_train(test_subjs)
    print('Train on: ', train_subjs, 'Test on: ', test_subjs)
    keras_train_and_save(train_subjs, test_subjs, params, load=True)
    test_finetuning(test_subjs, params)

def deg_to_pix(deg):
    posx, posy = deg
    dist_mm = 500.
    w_mm = 374.
    h_mm = 300.
    w_pix = 1280
    h_pix = 1024
    conv = lambda data, pix, mm, dist: \
        int(round(np.tan(data / 180. * np.pi) * dist * pix/mm + pix/2.))
    return conv(posx, w_pix, w_mm, dist_mm), \
        conv(-posy, h_pix, h_mm, dist_mm)

def calib_testing(subj, params):
    EET_DATA_ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\dataset\\Google project recordings\\Heatmaps_01_S_S{:03d}_R04_SHVSS3{:d}_BW_ML_120Hz\\'
    BLENDER_DATA_ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\psog_nn\\eet_data_converter\\blender_data\\{}\\{}_grid15.csv'

    subj_root = EET_DATA_ROOT.format(subj, 2 if subj < 11 else 4)

    subj = str(subj)

    data_path = 'DOT-R19.xml'
    for filename in os.listdir(subj_root):
        if filename.endswith('.xml'):
            data_path = filename

    tree = ET.parse(os.path.join(subj_root, data_path))
    root = tree.getroot()

    stimuli_pos = []
    for position in root.iter('Position'):
        x = int(position.find('X').text)
        y = int(position.find('Y').text)
        stimuli_pos.append((x, y))

    calib_pos = list(set(stimuli_pos))
    calib_pos.sort()
    calib_pos = [
        calib_pos[0],
        calib_pos[2],
        calib_pos[4],
        calib_pos[8],
        calib_pos[10],
        calib_pos[12],
        calib_pos[16],
        calib_pos[18],
        calib_pos[20]
    ]
    # stimuli_pos = list(set(stimuli_pos))
    # stimuli_pos.sort()

    stimuli_pos = np.array(stimuli_pos)

    X_train, y_train = get_train_data([subj])

    train_ind = []
    test_ind = []
    for ind, pos in enumerate(y_train):
        posx, posy = deg_to_pix(pos)
        calib_point = False
        for calib in calib_pos:
            x, y = calib
            dist = np.hypot(posx - x, posy - y)
            if dist < 35.:
                calib_point = True
                break
        if calib_point:
            train_ind.extend([ind])
        else:
            test_ind.extend([ind])
    
    X_test = X_train[test_ind]
    y_test = y_train[test_ind]
    
    X_train = X_train[train_ind]
    y_train = y_train[train_ind]

    X_train, X_test = normalize(X_train, X_test, subj, False, 'cnn')

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

    X_train = X_train.reshape((X_train.shape[0], 3, 5, 1))
    X_val = X_val.reshape((X_val.shape[0], 3, 5, 1))
    X_test = X_test.reshape((X_test.shape[0], 3, 5, 1))

    fit_time = time.time()
    model = train_keras_model(X_train, y_train, *params, X_val, y_val)
    fit_time = time.time() - fit_time

    train_acc = calc_acc(y_train, model.predict(X_train))
    print('Train acc: ', train_acc)
    test_acc = calc_acc(y_test, model.predict(X_test))
    print('Test acc: ', test_acc)

if __name__ == "__main__":
    # BEST_MLP_MODEL = (0, 0, 4, 96)
    # BEST_CNN_MODEL = (1, 4, 3, 96)
    # BEST_MLP_MODEL = (0, 0, 6, 20)
    BEST_CNN_MODEL = (4, 4, 4, 20)

    # cross_testing(['1', '2', '3', '4'], BEST_MLP_MODEL)
    # cross_testing(['5', '6', '7', '8'], BEST_MLP_MODEL)
    # cross_testing(['10', '11', '12', '13'], BEST_MLP_MODEL)
    # cross_testing(['14', '15', '16', '17'], BEST_MLP_MODEL)
    # cross_testing(['18', '19', '20'], BEST_MLP_MODEL)
    # cross_testing(['21', '22', '23'], BEST_MLP_MODEL)

    cross_testing(['1', '2', '3', '4'], BEST_CNN_MODEL)
    cross_testing(['5', '6', '7', '8'], BEST_CNN_MODEL)
    cross_testing(['10', '11', '12', '13'], BEST_CNN_MODEL)
    cross_testing(['14', '15', '16', '17'], BEST_CNN_MODEL)
    cross_testing(['18', '19', '20'], BEST_CNN_MODEL)
    cross_testing(['21', '22', '23'], BEST_CNN_MODEL)

    # calib_testing(8, BEST_CNN_MODEL)