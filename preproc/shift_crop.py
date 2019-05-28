""" Combine all shifts and crop to close eye caption.
"""
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from skimage.io import imread, imsave

from preproc.utils import get_default_shifts
from utils.gens import ImgPathGenerator
from utils.utils import repeat_up_to, calc_pad_size, do_sufficient_pad


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
    return int(round(sh / STEP)) * PIX_TO_MM

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
    top_lefts = (x - h // 2, y - w // 2)

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
            'GazePointXLeft': 'pos_x',
            'GazePointYLeft': 'pos_y',
            'PupilArea': 'pupil_size'
        }
    )
    data = data[['time', 'pos_x', 'pos_y', 'pupil_size']]

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

    output_dir = os.path.join(subj_root, 'images_crop')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(img_paths.get_root(), "_CropPos.txt")) as f:
        cp_x, cp_y = map(int, f.readline().split(' '))

    data = pd.read_csv(
        os.path.join(subj_root, 'FullSignal.csv'),
        sep='\t'
    )

    data = rename_subset(data)

    shifts = get_default_shifts(data.dropna().shape[0])
    data = add_sensor_shifts(data, shifts)

    with open(os.path.join(img_paths.get_root(), 'head_mov.txt'), 'r') as file:
        head_mov_data = [
            tuple(map(int, line.split(' ')))
            for line in file.readlines()
        ]

    img_ind = 0
    for i, img_path in enumerate(img_paths):
        if i >= data.shape[0] or data.iloc[i].isna().any():
            continue
        img = imread(img_path)
        img = get_shifted_crop(
            img,
            (cp_x, cp_y),
            head_mov_data[i],
            data.iloc[i]
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
    for dirname in os.listdir(dataset_root):
        if subj_ids is not None and dirname not in subj_ids:
            continue
        subj_root = os.path.join(dataset_root, dirname)
        shift_and_crop_subj(subj_root)

if __name__ == "__main__":
    shift_and_crop(sys.argv[1])
