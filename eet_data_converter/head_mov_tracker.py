import os

import cv2
import numpy as np
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.io import imread


def track_marker(img):
    coords = corner_peaks(corner_harris(img), min_distance=5)
    coords_subpix = corner_subpix(img, coords, window_size=13)

    max_resp = None
    for coord in coords_subpix:
        x, y = coord
        if np.isnan(x) or np.isnan(y):
            continue
        x = int(round(x))
        y = int(round(y))
        resp = kernel_response(img, x, y) 
        if max_resp is None or resp > max_resp:
            max_resp = resp
            max_x, max_y = x, y
    return max_x, max_y


def tracker_marker_cv(img):
    coords = cv2.goodFeaturesToTrack(img, 3, 0.1, 10)
    max_resp = None
    for coord in coords:
        x, y = coord[0]
        if np.isnan(x) or np.isnan(y):
            continue
        x = int(round(x))
        y = int(round(y))
        resp = kernel_response(img, x, y) 
        print(x, y, resp)
        if max_resp is None or resp > max_resp:
            max_resp = resp
            max_x, max_y = x, y
    return max_x, max_y


def init_pattern(img, x, y):
    return np.array(img[x - 4: x + 5, y - 5: y + 6], dtype=float)


def tracker_marker_eet(img, start_x=134, start_y=519, pattern=None):
    coords = corner_peaks(corner_harris(img), min_distance=5)
    coords_subpix = corner_subpix(img, coords, window_size=13)

    min_dist = None
    #for coord in coords_subpix:
        #x, y = coord
    for x in range(start_x - 15, start_x + 15 + 1):
        for y in range(start_y - 15, start_y + 15 + 1):
            if np.isnan(x) or np.isnan(y):
                continue
            x = int(round(x))
            y = int(round(y))
            dist = pattern_dist(img, x, y, pattern) 
            #print(x, y, dist)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_x, min_y = x, y
    return min_x, min_y

def pattern_dist(img, x, y, pattern=None):
    #print(np.array(img[x - 4: x + 6, y - 5: y + 6] > 0.55, dtype=float))
    if pattern is None:
        pattern = np.array([
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        ])
    return np.linalg.norm(
        np.array(img[x - 4: x + 5, y - 5: y + 6], dtype=float).flatten() -
        pattern.flatten(), ord=2
    )


def kernel_response(img, x, y):
    kernel = np.array([
        [-1., -1., -1., 1., 1.],
        [-1., -1., -1., 1., 1.],
        [-1., -1., -1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]
    ])
    resp = 0.
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            resp += img[y - 2 + i, x - 2 + j] / 255. * kernel[i, j]
    return resp


EET_DATA_ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\dataset\\Google project recordings\\Heatmaps_01_S_S{:03d}_R04_SHVSS3{:d}_BW_ML_120Hz\\'

if __name__ == "__main__":
    for subj in range(8, 9 + 1):
        subj_root = EET_DATA_ROOT.format(subj, 2 if subj < 11 else 4)

        print("Working dir: " + subj_root)

        input_dir = os.path.join(subj_root, 'vog_cnn')

        with open(os.path.join(input_dir, "_mc.txt")) as f:
            line = f.readline()
            start_x, start_y = map(int, line.split(' '))
            mark_x = start_x
            mark_y = start_y
            mark_pattern = None

        head_mov_file = open(os.path.join(input_dir, "head_mov.txt"), 'w')

        for filename in os.listdir(input_dir):
            if not filename.endswith('.jpg'):
                continue

            sk_img = imread(os.path.join(input_dir, filename), as_gray=True)
            if mark_pattern is None:
                mark_pattern = init_pattern(sk_img, mark_x, mark_y)
            mark_x, mark_y = tracker_marker_eet(sk_img, mark_x, mark_y, mark_pattern)

            img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
            print(start_x - mark_x, start_y - mark_y, file=head_mov_file)
            
            #cv2.circle(img, (mark_y, mark_x), 3, (255,0,255), 2)
            #cv2.imshow("eye", img)
            #if cv2.waitKey(1) == ord('q'):
            #    continue
            #    break


