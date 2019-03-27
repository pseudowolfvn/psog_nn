""" Basic training time complexity analysis.
"""
import sys

import numpy as np

from ml.eval import StudyEvaluation
from plots.utils import accumulate_data


def calc_stats(data, field):
    """Calculate statistic of mean and standard deviation for provided
        quantity of results.

    Args:
        data: A dict with results returned from StudyEvaluation.run().
        field: A string with measured quantity.
            'data' for spatial accuracy and
            'time' for time spent for training are supported.

    Returns:
        A dict with aforementioned statistics.
    """
    def arr(data, appr):
        return [data[subj][appr][field] for subj in data['subjs']]

    ft = arr(data, 'ft')
    scr = arr(data, 'scr')
    stats = {
        'ft': {
            'mean': np.mean(ft),
            'std': np.std(ft),
        },
        'scr': {
            'mean': np.mean(scr),
            'std': np.std(scr),
        }
    }
    return stats

def evaluate_config(eval, config):
    """Run the training time evaluation for provided training configuration.

    Args:
        eval: A properly initialized instance of ml.eval.StudyEvaluation class.
        config: A dict with training configuration following the format of
            StudyEvaluation.run().
    """
    data = eval.run(config, 1)
    time_stats = calc_stats(data, 'time')
    acc_stats = calc_stats(data, 'data')
    print('Time statistics for config: ', config)
    print(time_stats)
    print('Accuracy statistics for config: ', config)
    print(acc_stats)
    print()

def evaluate_time(root, archs, setups, redo=True):
    """Run the training time evaluation for the whole dataset.

    Args:
        root: A string with path to dataset.
        archs: A list with neural network architectures to evaluate.
        setups: A list with power consumption setups to evaluate.
        redo: A boolean that shows if evaluation should be done again
            if files of results already exist.
    """
    eval = StudyEvaluation(root, archs, setups, 'time', redo=redo)
    # smaller batch size
    sm_bs_config = {
        'batch_size': 200,
        'patience': 50
    }
    evaluate_config(eval, sm_bs_config)
    # bigger batch size
    bg_bs_config = {
        'batch_size': 2000,
        'patience': 50
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
    model.fit(X_train, y_train, nb_epoch=1000, batch_size=200
            , validation_data=(X_val, y_val), verbose=1
            , callbacks=[early_stopping])
    fit_time = time.time() - fit_time

    train_acc = calc_acc(y_train, model.predict(X_train))
    print('Train acc: ', train_acc)
    test_acc = calc_acc(y_test, model.predict(X_test))
    print('Test acc: ', test_acc)
    return train_acc, test_acc, fit_time

if __name__ == "__main__":
    # evaluate_time(sys.argv[1], ['cnn'], ['lp'], redo=False)
    # exit()
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
    # _, _, t = load_and_finetune(
    #     sys.argv[1], train_subjs, '23', get_best_model_params('cnn', 'lp'), sm_bs_config
    # )

    # TRAIN FROM SCRATCH, ~20 sec
    _, _, t = train_from_scratch(
            sys.argv[1], '23', get_best_model_params('cnn', 'lp'), sm_bs_config
    )

    print('Time: ', t)