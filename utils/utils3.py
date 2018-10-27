#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:26:01 2017

@author: raimondas
"""
#%% imports
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA as pca

import itertools, operator
import json

#%% constants


#%% functions
def rgb2gray(rgb):
    '''simple rgb to gray conversion'''
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



def record_exists(results, _n_calib, _n_split, _params):
    _params_str = json.dumps(_params)
    _r = results.query("n_calib == %d and \
                   split == %d and \
                   params == '%s'" % (_n_calib, _n_split, _params_str))
    return True if len(_r) else False

def create_calib_scheme(fname_config_calib):
    #manually create calibration config. Couple of examples. Use one
    config_calib = {
        'training': [0, 1, 2, 5, 6, 7, 10, 11, 12],
        'validation': [3, 4, 8, 9]
    }
    config_calib = {
        'training': [0, 6, 12, 10, 2],
        'validation': [1, 3, 4, 5, 7, 8, 9, 11]
    }
    config_calib = {
        'training': range(13),
        'validation': [6]
    }
    with open(fname_config_calib, 'w') as f:
        json.dump(config_calib, f, indent=4)

def get_px2deg(geom):
    """Calculates pix2deg values, based on simple geometry.
    Parameters:
        geom    --  dictionary with following parameters of setup geometry:
                    screen_width
                    screen_height
                    eye_distance
                    display_width_pix
                    display_height_pix
    Returns:
        px2deg  --  pixels per degree value
    """
    px2deg = np.mean(
        (1/
         (np.degrees(2*np.arctan(geom['screen_width']/
                    (2*geom['eye_distance'])))/
         geom['display_width_pix']),
         1/
         (np.degrees(2*np.arctan(geom['screen_height']/
                    (2*geom['eye_distance'])))/
         geom['display_height_pix']))
    )
    return px2deg

def calc_etdq(data_gt, data_pr, fixations_index, fmask=None):
    """Calculates eye tracking data quality measures.
    parameters:

        fmask   --  mask to select only certain fixations
                    from which accuracy is calculated
    """
#    ##debug
#    fixations_index = fixations_manual
#    fmask=None
#    data_gt = y_exp
#    data_pr = pred_exp
#    ##

    fmask = np.ones(len(fixations_index)).astype(np.bool) if fmask is None\
                                                          else fmask

    #extract gaze positions for each selected fixation
    fi = [(_i, _n) for _n, (_s, _e) in enumerate(fixations_index) \
                   #for _i in range(_s, _e+1)] #does not account for 1 sample
                   for _i in range(_s-1, _e)] #adjusts for matlab indexing
    i, n = zip(*fi)
    etdq_df = pd.DataFrame(np.concatenate([np.hstack([data_gt, data_pr])[list(i)],
                                           np.array(n).reshape(-1,1)], axis=1),
                           columns = ['gt_x', 'gt_y', 'pr_x', 'pr_y', 'fix_ind'])



    ##acc
    #calculate fixation position, aka average of samples in that fixation
    acc = etdq_df.groupby('fix_ind', as_index=False).apply(np.mean)
    acc = acc[fmask]

#    ##debug
#    plt.plot(data_gt)
#    plt.plot(data_pr)
#    for _fix_ind, _f in etdq_df.groupby('fix_ind', as_index=False):
#        mask = n==_fix_ind
#        r = np.array(i)[mask]
#        plt.plot(r, _f['gt_x'], 'grey')
#        plt.plot(r, _f['pr_x'], 'grey')
#        plt.plot(r, list(itertools.repeat(np.mean(_f['gt_x']), len(r))), 'k')
#        plt.plot(r, list(itertools.repeat(np.mean(_f['pr_x']), len(r))), 'k')
#
#    ##

    acc_l2 = np.hypot(np.diff(acc[['pr_x', 'gt_x']].values, axis=1),
                      np.diff(acc[['pr_y', 'gt_y']].values, axis=1)).mean()

    ##rms
    rms = []
    #iterate through fixations
    for _, _f in etdq_df.groupby('fix_ind', as_index=False):
        _d = _f.diff(axis=0)[1:] #differentiate
        #could be optimized, because takes sqrt and then squares again
        _rms_gt = np.sqrt(np.mean(np.hypot(*zip(*_d[['gt_x', 'gt_y']].values))**2))
        _rms_pr = np.sqrt(np.mean(np.hypot(*zip(*_d[['pr_x', 'pr_y']].values))**2))
        rms.append([_rms_gt, _rms_pr])
    rms = np.array(rms)[fmask]
    rms = rms.mean(axis=0)

    return acc_l2, rms

def calc_nparams(design, nn_params, decode=True, n_outputs=2):
#    ###debug
#    design=DESIGN
#    nn_params=_params
#    decode=False
#    ###
    _nn_params = json.loads(nn_params) if decode else nn_params
    _n_neurons = list(_nn_params['hidden_layer_sizes'])

    with open('designs/%s.json'%design, 'r') as f:
        design = json.load(f)
        n_input = len(design['output']) if len(design['output'])\
                                        else len(design['design'])

    architecture = [n_input] + _n_neurons + [n_outputs]
    n_params = sum(itertools.imap(operator.mul,
                                  architecture[:-1],
                                  architecture[1:])) + \
                                  sum(_n_neurons)+n_outputs #biases
    return n_params

class UnitScaler():
    def __init__(self):
        pass
    def fit(self, x):
        pass
    def transform(self, x):
        return x

class Clf():
    """Improved interface to sklearn classifiers.
    """
    def __init__(self, clf, sep=False):
        self.clf = clf
        #self.normalizer = preprocessing.StandardScaler()
        #self.normalizer = UnitScaler()

        self.normalizer = pca(n_components=0.99, whiten=True, random_state=0o62217)
        #self.normalizer = pca(whiten=True, random_state=062217)

        self.sep = sep
    def fit(self, X, y):
        """
        y must be 2-dimensional vector
        """
        if not(np.ndim(y) == 2):
            raise "Labels must be 2-d vector"

        #interface for other ML algorithms. Builds two separate classifiers
        if self.sep:
            self.clf_h = copy.deepcopy(self.clf)
            self.clf_v = copy.deepcopy(self.clf)
            self.clf_h.fit(X, y[:,0])
            self.clf_v.fit(X, y[:,1])
        #interface for MLP
        else:
            self.clf.fit(X, y)

    def predict(self, X_test):
        #interface for other ML algorithms
        if self.sep:
            pred_h = self.clf_h.predict(X_test)
            pred_v = self.clf_v.predict(X_test)
            pred = np.vstack((pred_h, pred_v)).T
        #interface for MLP
        else:
            pred = self.clf.predict(X_test)
        return pred

    def get_params(self, deep=True):
        return self.clf.get_params(deep=deep)

    def run_training(self, X, y, train_index=None, test_index=None):
        if train_index is not None:
            X_train, y_train = X[train_index], y[train_index]
        else:
            X_train, y_train = X, y

        if test_index is not None:
            X_test, y_test = X[test_index], y[test_index]
        else:
            X_test, y_test = X, y

        #normalise
        self.normalizer.fit(X_train)
        X_train_ = self.normalizer.transform(X_train)
        X_test_ = self.normalizer.transform(X_test)

        #fit
        self.fit(X_train_, y_train)

        #predict
        pred_train = self.predict(X_train_)
        pred_test = self.predict(X_test_)

        #MAE for horizontal and vertical channels
        mae_train = metrics.mean_absolute_error(y_train, pred_train,
                                                multioutput='raw_values')
        mae_test = metrics.mean_absolute_error(y_test, pred_test,
                                               multioutput='raw_values')

        #Euclidian MAE
        err_train = y_train - pred_train
        mae_train_l2 = np.mean(np.hypot(err_train[:,0], err_train[:,1]))

        err_test = y_test - pred_test
        mae_test_l2 = np.mean(np.hypot(err_test[:,0], err_test[:,1]))
        return (pred_train, pred_test), \
               (mae_train, mae_test), \
               (mae_train_l2, mae_test_l2)

    def run_infer(self, X, y=None):
        pred = self.predict(self.normalizer.transform(X))

        #MAE for horizontal and vertical channels
        mae = metrics.mean_absolute_error(y, pred, multioutput='raw_values') \
              if y is not None else None

        #Euclidian MAE
        err = y - pred
        mae_l2 = np.mean(np.hypot(err[:,0], err[:,1]))

        return pred, mae, mae_l2