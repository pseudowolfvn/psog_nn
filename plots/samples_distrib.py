""" Plot per subject stimulus samples distribution.
"""
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from ml.load_data import get_subj_data, get_stimuli_pos, get_calib_like_data
from ml.utils import normalize
from plots.utils import get_module_prefix
from utils.utils import deg_to_pix


# TODO: rewrite with use of ml.load_data.get_calib_like_data()
def draw_subj_samples(root, subj):
    """Draw stimuli and eye movements positions in pixels with highlighting
        points for calibration-like distribution training set
        simulation for provided subject.

    Args:
        root: A string with path to dataset.
        subj: A string with specific subject id.
    """
    subj_root = os.path.join(root, subj)
    print('Drawing samples distribution for subj:', subj_root)
    
    stimuli_pos = get_stimuli_pos(root, subj)

    stimuli_grid = sorted(list(set(stimuli_pos)))
    stimuli_pos = np.array(stimuli_pos)

    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_calib_like_data(root, subj, 'cnn')

    h, w = 1024, 1280
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = [255, 255, 255]

    print('Train: ', y_train.shape)
    for pos in y_train:
        x, y = deg_to_pix(pos)
        cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

    # for pos in y_val:
    #     x, y = deg_to_pix(pos)
    #     cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    print('Test: ', y_test.shape)
    for pos in y_test:
        x, y = deg_to_pix(pos)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    # workaround for correct order
    grid_order = np.array([
        1, 6, 9, 14, 17,
        2, 10, 18,
        3, 7, 11, 15, 19,
        4, 12, 20,
        5, 8, 13, 16, 21]) - 1
    for ind, grid_ind in enumerate(grid_order):
        pos = stimuli_grid[grid_ind]
        x, y = pos
        cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
        cv2.putText(
            img, str(ind + 1), (x - 7, y - 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2
        )
        if ind + 1 in [1, 3, 5, 9, 11, 13, 17, 19, 21]:
            cv2.circle(img, (x, y), 35, (0, 0, 0), 1)

    img_dir = os.path.join(get_module_prefix(), 'samples_distrib')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    cv2.imwrite(os.path.join(img_dir, str(subj) + '.jpg'), img)

def draw_samples(root, subj_ids=None):
    """Draw stimuli and eye movements positions in pixels with highlighting
        points for calibration-like distribution training set simulation.

    Args:
        root: A string with path to dataset.
        subj_ids: A list with subjects ids to draw for if provided,
            otherwise draw for the whole dataset.
    """
    for dirname in os.listdir(root):
        if subj_ids is not None and dirname not in subj_ids:
            continue
        draw_subj_samples(root, dirname)

if __name__ == "__main__":
    draw_samples(sys.argv[1], subj_ids=['1'])
