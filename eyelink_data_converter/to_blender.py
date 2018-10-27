import os

import numpy as np
import pandas as pd


from .filtration import filtration, discard_missing
from utils.metrics import calc_acc
from utils.utils import get_eye_stim_signal


def gen_sensor_shifts(data, hor, ver):
    hor = np.ravel(hor)
    ver = np.ravel(ver)
    N = len(data.index)
    S = hor.shape[0] * ver.shape[0]
    shifts = np.zeros((N, 2))
    shift_pairs = np.zeros((S, 2))
    s = 0
    for h in hor:
        for v in ver:
            shift_pairs[s] = h, v
            s += 1
    s = 0
    for n in range(N):
        shifts[n, :] = shift_pairs[s]
        s = (s + 1) % S

    data['hor_shift'] = shifts[:,0] 
    data['ver_shift'] = shifts[:,1]
    data['dep_shift'] = np.zeros((N, 1))
    return data


def rename_from_eyelink(eyelink_data):
    data = eyelink_data.rename(index=str, columns={
        'X_Position_Degree_Left_Eye': 'eye_x',
        'Y_Position_Degree_Left_Eye': 'eye_y',
        'Validity_Left_Eye': 'valid',
        'Pupil_Diameter_LeftEye': 'pupil',
        'Target X': 'stim_x',
        'Target Y': 'stim_y'
    })
    return data


def rename_to_blender(data):
    blender_data = data.drop(columns=['stim_x', 'stim_y', 'acc'
        , 'radial_eye', 'radial_stim'])
    blender_data = blender_data.rename(index=str, columns={
        'eye_x': 'posx',
        'eye_y': 'posy',
        'hor_shift': 'smh',
        'dep_shift': 'smd',
        'ver_shift': 'smv',
        'pupil': 'pupil_size'
    })
    blender_data = blender_data[['posx', 'posy'
        , 'smh', 'smd', 'smv', 'pupil_size']]

    blender_data['pupil_size'] /= 1000.
    
    return blender_data

EYELINK_DATA_ROOT = r'.\eyelink_data_converter\eyelink_data'
BLENDER_DATA_ROOT = r'.\eyelink_data_converter\blender_data'

def convert_to_blender():
    i = 1
    for filename in os.listdir(EYELINK_DATA_ROOT):
        if not filename.endswith('.asc'):
            continue

        output_path = os.path.join(BLENDER_DATA_ROOT, str(i))

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        log_file = open(os.path.join(output_path, str(i) + '.log'), 'w')

        print('Converting: ', filename, file=log_file)
        eyelink_data = pd.read_csv(os.path.join(EYELINK_DATA_ROOT, filename)
            , sep='\t')

        data = rename_from_eyelink(eyelink_data)

        # discard invalid data
        data = discard_missing(data)

        # extract monocular info 
        data = data[['eye_x', 'eye_y',
            'pupil', 'stim_x', 'stim_y']]

        print('Spatial accuracy before filtration: '
            , calc_acc(*get_eye_stim_signal(data)), file=log_file)

        data = filtration(data)

        print('Spatial accuracy after filtration: '
            , calc_acc(*get_eye_stim_signal(data)), file=log_file)

        data = gen_sensor_shifts(data, 
            [-2.0, -1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5, 2.0],
            [-2.0, -1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5, 2.0])

        blender_data = rename_to_blender(data)

        blender_data.to_csv(os.path.join(output_path, str(i) + '.csv')
            , sep='\t', index=False)

        print(blender_data)
        i += 1
        # uncomment this for testing purposes!
        # break