""" Plot per subject stimulus samples distribution.
"""
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from ml.load_data import get_subj_data, normalize
from plots.utils import get_module_prefix
from utils.utils import deg_to_pix

def draw_subj_samples(subj_root):
    print('Drawing samples distribution for subj:', subj_root)
    subj = Path(subj_root).name

    data_path = 'Stimulus.xml'

    tree = ET.parse(os.path.join(subj_root, data_path))
    root = tree.getroot()

    stimuli_pos = []
    for position in root.iter('Position'):
        x = int(position.find('X').text)
        y = int(position.find('Y').text)
        stimuli_pos.append((x, y))

    calib_pos = sorted(list(set(stimuli_pos)))
    calib_pos = [
        calib_pos[0],
        calib_pos[2],
        calib_pos[4],
        calib_pos[8],
        calib_pos[10],
        calib_pos[12],
        calib_pos[16],
        calib_pos[18],
        calib_pos[20]
    ]

    stimuli_grid = sorted(list(set(stimuli_pos)))
    stimuli_pos = np.array(stimuli_pos)

    X_train, y_train = get_subj_data(subj_root)

    train_ind = []
    test_ind = []
    for ind, pos in enumerate(y_train):
        posx, posy = deg_to_pix(pos)
        calib_point = False
        for calib in calib_pos:
            x, y = calib
            dist = np.hypot(posx - x, posy - y)
            if dist < 35.:
                calib_point = True
                break
        if calib_point:
            train_ind.extend([ind])
        else:
            test_ind.extend([ind])

    X_test = X_train[test_ind]
    y_test = y_train[test_ind]

    X_train = X_train[train_ind]
    y_train = y_train[train_ind]

    X_train, X_test = normalize(X_train, X_test, subj, False, 'cnn')

    X_test, _, y_test, _ = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42)

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
    for dirname in os.listdir(root):
        if subj_ids is not None and dirname not in subj_ids:
            continue
        subj_root = os.path.join(root, dirname)
        draw_subj_samples(subj_root)

if __name__ == "__main__":
    draw_samples(sys.argv[1])
