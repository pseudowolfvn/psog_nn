""" Combine all shifts and crop to close eye caption.
"""
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skimage.io import imread, imsave

from preproc.utils import get_default_shifts, get_randn_shifts, get_no_shifts
from utils.gens import ImgPathGenerator
from utils.utils import repeat_up_to, calc_pad_size, \
    do_sufficient_pad, find_record_dir


def add_sensor_shifts(data, shifts):
    """Add columns with array of provided simulated sensor shifts.

    Args:
        data: A pandas DataFrame with eye-movement signal.
        shifts: A numpy array of (N, 2) shape where
            columns represent horizontal and vertical shifts.

    Returns:
        A DataFrame with added columns.
    """
    data.loc[data.dropna().index, 'sh_hor'] = shifts[:, 0]
    data.loc[data.dropna().index, 'sh_ver'] = shifts[:, 1]
    return data

def shift_mm_to_pix(sh):
    """Convert provided shift in millimeters (mm) to pixels.

    Args:
        sh: A float with shift in mm.

    Returns:
        A float with corresponding shift in pixels.
    """
    STEP = 0.5
    PIX_TO_MM = 4
    return sh / STEP * PIX_TO_MM

def get_valid_sample(data, i):
    start_i = i
    while i >= 0 and data.iloc[i].val == 4:
        i -= 1
    if i < 0:
        i = start_i
        while data.iloc[i].val == 4:
            i += 1
    return data.iloc[i]

def build_head_mov_correction_func(data):
    def get_regression_coefs(X, y):
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        return reg.coef_[0][0], reg.intercept_[0]
    
    A_x, _ = get_regression_coefs(data.pos_x.values, data.pc_x.values)
    A_y, _ = get_regression_coefs(data.pos_y.values, data.pc_y.values)

    return lambda pos_x, pos_y: (-A_x*pos_x, -A_y*pos_y)


def get_shifted_crop(img, center, head_mov, sample):
    """Crop provided image to 320x240 close eye capture correcting for
        possible head movement and introducing corresponding sensor shift.

    Args:
        img: An image of type convertible to numpy array.
        center: A tuple with vertical, horizontal coordinates
            of rectangle crop center.
        head_mov: A tuple with vertical, horizontal
            components of head movement.
        sample: A dict-like object with vertical, horizontal
            components of sensor shift that are accessible via
            'sh_ver' and 'sh_hor' fields respectively.

    Returns:
        A cropped image.
    """
    x, y = center
    x += shift_mm_to_pix(sample['sh_ver']) + head_mov[0]
    y += shift_mm_to_pix(sample['sh_hor']) + head_mov[1]

    w, h = 320, 240
    top_lefts = tuple(
        map(
            lambda z: int(round(z)),
            (x - h / 2, y - w / 2)
        )
    )

    top_lefts, pad = calc_pad_size(img, top_lefts, (h, w))
    img = do_sufficient_pad(img, pad)

    x, y = top_lefts
    return img[x: x + h, y: y + w]

def rename_subset(data):
    """Leave the subset of data with
    timestamp, horizontal and vertical parts of signal, pupil size
    and rename corresponding columns for more handy access.

    Args:
        data: A pandas DataFrame with eye-movement signal following
            the format of subject's FullSignal.csv from ETRA 2019 dataset.

    Returns:
        A DataFrame with renamed columns subset.
    """
    data = data.rename(
        index=str,
        columns={
            'Timestamp': 'time',
            'ValidityLeft': 'val',
            'GazePointXLeft': 'pos_x',
            'GazePointYLeft': 'pos_y',
            'PupilLeftX': 'pc_x',
            'PupilLeftY': 'pc_y',
            'PupilArea': 'pupil_size'
        }
    )
    data = data[['time', 'val', 'pos_x', 'pos_y', 'pc_x', 'pc_y', 'pupil_size']]

    return data

def shift_and_crop_subj(subj_root):
    """Introduce artificial sensor shift and crop to close eye capture
        images in the whole recording for provided subject.

    Args:
        subj_root: A string with full path to directory
            with subject's recording stored in images.
    """
    print("Shift and crop for subject: " + subj_root)

    img_paths = ImgPathGenerator(subj_root)

    output_dir = os.path.join(subj_root, 'images_crop_noshift')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # with open(os.path.join(img_paths.get_root(), "_CropPos.txt")) as f:
    #     cp_x, cp_y = map(int, f.readline().split(' '))

    data_name = 'FullSignal.csv'
    for filename in os.listdir(subj_root):
        if filename.startswith('DOT') and filename.endswith('.tsv'):
            data_name = filename

    data = pd.read_csv(
        os.path.join(subj_root, data_name),
        sep='\t'
    )

    valid_data = data[data.ValidityLeft != 4]

    data = rename_subset(data)

    shifts = get_no_shifts(data.dropna().shape[0])
    data = add_sensor_shifts(data, shifts)

    # with open(os.path.join(img_paths.get_root(), 'head_mov.txt'), 'r') as file:
    #     head_mov_data = [
    #         tuple(map(int, line.split(' ')))
    #         for line in file.readlines()
    #     ]
    head_mov_func = build_head_mov_correction_func(data)

    img_ind = 0
    for i, img_path in enumerate(img_paths):
        if i >= data.shape[0] or data.iloc[i].isna().any():
            continue
        img = imread(img_path)

        sample = get_valid_sample(data, i)

        cp_x, cp_y = sample.pc_x, sample.pc_y
        head_mov_x, head_mov_y = head_mov_func(sample.pos_x, sample.pos_y)

        img = get_shifted_crop(
            img,
            (cp_y, cp_x),
            (head_mov_y, head_mov_x),
            sample
        )

        # TODO: at the next step the image-wise map between
        # the full signal and with dropped missed samples is lost
        img_name = str(img_ind) + '.jpg'
        imsave(os.path.join(output_dir, img_name), img)
        img_ind += 1

    # TODO: truncate either data or images to have the same amount of each

    data_name = Path(subj_root).name + '.csv'
    data.dropna().to_csv(
        os.path.join(output_dir, data_name),
        sep='\t',
        index=False
    )


def shift_and_crop(dataset_root, subj_ids=None):
    """Introduce artificial sensor shift and crop to close eye capture
        images for the provided list of subjects.

    Args:
        dataset_root: A string with path to dataset.
        subj_ids: A list with subjects ids to shift and crop for if provided,
            otherwise shift and crop for the whole dataset.
    """
    subj_dirs = []
    if subj_ids is not None:
        subj_dirs = [
            find_record_dir(dataset_root, subj_id) for subj_id in subj_ids
        ]

    for dirname in os.listdir(dataset_root):
        if (subj_ids is not None and dirname not in subj_dirs) or \
                not dirname.startswith('Record'):
            print('Skipping', dirname, '...')
            continue

        subj_root = os.path.join(dataset_root, dirname)
        shift_and_crop_subj(subj_root)

if __name__ == "__main__":
    shift_and_crop(sys.argv[1])
