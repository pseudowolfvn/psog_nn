import os
import sys

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from em_classification.calc_utils import enrich_data
from em_classification.data_utils import convert_to_window, get_model_path, \
    get_subj_data, scale_data, to_channel_first
from em_classification.vog.ivt import IVT
from em_classification.psog.nn import Model1DCNN
from ml.calib_data import get_stim_pos_subj_data, get_subj_calib_data
from preproc.psog import PSOG
from utils.utils import find_filename, find_record_dir


def classify_vog(root, subj_id, data_id):
    gaze_data = get_subj_data(root, subj_id, data_id, include_outliers=False)
    stim_data = get_stim_pos_subj_data(root, subj_id)

    ivt = IVT(gaze_data, stim_data, EPS=0.4)
    _, all_merged_fix = ivt.get_calib_fixations()

    # get_subj_calib_data from ml.calib_data now requires
    # data in the format that already has classified data
    # TODO: rewrite to use coherent data format
    # so the get_subj_calib_data from ml.calib_data can be reused
    data = get_subj_data(root, subj_id, data_id, include_outliers=True)

    data['fixation'] = 0
    first_fix_beg = all_merged_fix[0][0]
    last_fix_end = all_merged_fix[-1][1]
    data.loc[data.time < first_fix_beg, 'fixation'] = -1
    data.loc[data.time > last_fix_end, 'fixation'] = -1
    for fix in all_merged_fix:
        beg, end = fix
        data.loc[(data.time >= beg) & (data.time <= end), 'fixation'] = 1

    return data

def classify_psog(data):
    X_cols = PSOG().get_names()
    y_cols = ['fixation']

    X = data[X_cols].values.astype(np.float32)
    y = data[y_cols].values

    X = scale_data(X)

    W = 11
    X, y = convert_to_window(X, y, w=W)

    [X] = to_channel_first([X])

    model_params = {
        'L_conv': 3,
        'K': 5,
        'L_fc': 5,
        'N': 20,
        'p_drop': 0.1,
        'ch_in': X.shape[1],
        'l_in': X.shape[2]
    }

    model = Model1DCNN(**model_params)

    model.load_weights(get_model_path())

    y_pred = model.predict(X)
    pad = np.repeat(-1, W)
    y_pred = np.concatenate((pad, y_pred, pad))

    data['prediction'] = y_pred
    # data.loc[data.fixation != -1, 'prediction'] = y_pred

    return data

def save_classified_data(root, subj_id, data_id, data):
    subj_root = os.path.join(root, subj_dir)
    data_name = subj_id + '_' + data_id + '.csv'
    data_path = os.path.join(subj_root, data_name)

    data.to_csv(
        data_path, sep='\t', index=False
    )

if __name__ == '__main__':
    root = sys.argv[1]
    data_id = sys.argv[2]

    for i in range(4, 4 + 1):
        subj_id = str(i)
        subj_dir = find_record_dir(root, subj_id)
        subj_root = os.path.join(root, subj_dir)

        print('Classifying VOG gaze signal using I-VT for: ', subj_root)
        data = classify_vog(root, subj_id, data_id)

        print('Classifying PSOG gaze signal using NN for: ', subj_root)
        data = classify_psog(data)

        save_classified_data(root, subj_id, data_id, data)
