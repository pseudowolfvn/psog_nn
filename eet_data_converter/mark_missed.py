import os

import numpy as np
from scipy.io import loadmat

from utils.utils import mat_to_df


ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\dataset\\Google project recordings\\Heatmaps_01_S_S{:03d}_R04_SHVSS3{:d}_BW_ML_120Hz\\'

for subj in range(1, 23 + 1):
    subj_root = ROOT.format(subj, 2 if subj < 11 else 4)

    print("Working dir: " + subj_root)

    imgs_path = os.path.join(subj_root, 'images')

    mat_path = 'Recording_InstanteneousVelocity.mat'
    mat = loadmat(os.path.join(subj_root, mat_path))
    data = mat['Recording']
    for ind, sample in enumerate(data):
        if np.isnan(sample[1]):
            old_name = os.path.join(imgs_path, str(ind) + '.jpg')
            new_name = os.path.join(imgs_path, str(ind) + '_NaN.jpg')
            print(old_name + ' to ' + new_name)
            os.rename(old_name, new_name)