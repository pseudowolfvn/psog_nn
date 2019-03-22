""" Basic training time complexity analysis.
"""
import sys

import numpy as np

from ml.eval import StudyEvaluation
from plots.utils import accumulate_data

def calc_time_stats(data):
    def time_arr(data, appr):
        return [data[subj][appr]['time'] for subj in data['subjs']]

    ft_time = time_arr(data, 'ft')
    scr_time = time_arr(data, 'scr')
    time_stats = {
        'ft': {
            'mean': np.mean(ft_time),
            'std': np.std(ft_time),
        },
        'scr': {
            'mean': np.mean(scr_time),
            'std': np.std(scr_time),
        }
    }
    return time_stats

def evaluate_config(eval, config):
    data = eval.run(config, 1)
    stats = calc_time_stats(data)
    print('Time statistics for config: ', config)
    print(stats)
    print()

def evaluate(root, archs, setups, redo=True):
    eval = StudyEvaluation(root, archs, setups, 'time', redo=redo)
    # smaller batch size
    sm_bs_config = {
        'batch_size': 200,
        'patience': 10
    }
    # evaluate_config(eval, sm_bs_config)
    # bigger batch size
    bg_bs_config = {
        'batch_size': 2000,
        'patience': 10
    }
    evaluate_config(eval, bg_bs_config)

def keras_load_and_finetune_fc(train_subjs, subj, params):
    import os
    import time

    from keras.models import load_model
    from keras.optimizers import Nadam
    from keras.callbacks import EarlyStopping

    from ml.load_data import get_specific_data
    from utils.metrics import calc_acc

    def get_model_name(train_subjs, params):
        return os.path.join(
            r'./ml/models',
            'keras_' + str(params) + '_' + str(train_subjs) + '.h5'
        )

    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_specific_data(sys.argv[1], subj, 'cnn', train_subjs)
    
    model = load_model(get_model_name(train_subjs, params))
    model.summary()
    for layer in model.layers:
        if layer.name.startswith('conv2d'):
            layer.trainable = False
    print('Model loaded')
    nadam_opt = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=nadam_opt)
    early_stopping = EarlyStopping(monitor='val_loss'
        , patience=50, mode='auto', restore_best_weights=True)

    print('X: ', X_train)
    print('y: ', y_train)
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

if __name__ == "__main__":
    evaluate(sys.argv[1], ['cnn'], ['lp'], redo=False)
    exit()
    from ml.from_scratch import train_from_scratch
    from ml.finetune import load_and_finetune
    from ml.grid_search import get_best_model_params
    from ml.load_data import split_test_from_all

    subjs_split = [
        ['1', '2', '3', '4'],
        ['5', '6', '7', '8'],
        ['9', '10', '11', '12'],
        ['13', '14', '15', '16'],
        ['17', '18', '19', '20'],
        ['21', '22', '23']
    ]

    train_subjs, test_subjs = split_test_from_all(subjs_split[-1])

    sm_bs_config = {
        'batch_size': 200,
        'patience': 50
    }

    bg_bs_config = {
        'batch_size': 2000,
        'patience': 50
    }

    # OLD FINE-TUNE, <10 sec
    # _, _, t =  keras_load_and_finetune_fc(train_subjs, '23', (2, 4, 4, 20))

    # REFACTORED FINE-TUNE, DOESN'T WORK, ~20sec
    _, _, t = load_and_finetune(
        sys.argv[1], train_subjs, '23', get_best_model_params('cnn', 'lp'), bg_bs_config
    )

    # TRAIN FROM SCRATCH, ~20 sec
    # _, _, t = train_from_scratch(
    #         sys.argv[1], '23', get_best_model_params('cnn', 'lp'), bg_bs_config
    # )

    print('Time: ', t)