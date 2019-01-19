import math
import os
import sys

import numpy as np
import pandas as pd


def fill_nans(data):
    # TODO: move all hardware specific stuff to a separate config
    # reported by recording hardware
    FPS = 120.040660581201
    # sampling frequency of the recording
    FREQ = 1000
    time_gap = FREQ/FPS
    samples_estimated = math.ceil(data['Timestamp'].iloc[-1] / FREQ * FPS)
    
    # storing the intermediate data in a numpy array is much more efficient
    np_full_data = np.zeros((samples_estimated, data.shape[1]))
    np_full_data[:] = np.nan

    j = 0
    for i in range(samples_estimated):
        # set estimated timestamp
        np_full_data[i][0] = round(i * time_gap)
        # TODO: ask Evgeniy about the logic behind
        if data.at[j, 'Timestamp'] - \
            np_full_data[i][0] <= 0.3 * time_gap:
            
            np_full_data[i][:] = data.iloc[j].values
            j += 1

    full_data = pd.DataFrame(
        np_full_data,
        columns=data.columns
    )
    full_data['Timestamp'] = full_data['Timestamp'].astype(np.int)
    return full_data

def restore_missed_samples(dataset_root):
    # TODO: think about a better name?
    INPUT_NAME = 'unprocessed_signal.csv'
    OUTPUT_NAME = 'full_signal.csv'
    for dirname in os.listdir(dataset_root):
        subj_root = os.path.join(dataset_root, dirname)
        data = pd.read_csv(
            os.path.join(subj_root, INPUT_NAME),
            sep='\t'
        )
        print('Restoring missed samples for subject: ', dirname)
        samples_before = data.shape[0]
        print('Samples before: ', samples_before, end='')

        data = fill_nans(data)
        data.to_csv(
            os.path.join(subj_root, OUTPUT_NAME),
            sep='\t',
            na_rep=np.nan,
            index=False
        )

        diff = data.shape[0] - samples_before
        print(', after: ', data.shape[0], ' (difference: ', diff, ')', sep='')

if __name__ == '__main__':
    restore_missed_samples(sys.argv[1])