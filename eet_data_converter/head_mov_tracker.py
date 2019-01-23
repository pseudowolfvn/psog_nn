import os
import sys

import numpy as np
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.io import imread

from utils.img_path_gen import ImgPathGenerator

def init_pattern(img, x, y):
    return np.array(img[x - 4: x + 5, y - 5: y + 6], dtype=float)

def pattern_dist(img, x, y, pattern):
    return np.linalg.norm(
        np.array(img[x - 4: x + 5, y - 5: y + 6], dtype=float).flatten() -
            pattern.flatten(),
        ord=2
    )

# TODO: this doesn't work and right now initial marker position is determined
# manually, come up with a better automatic algorithm
def find_marker(img):
    min_dist = np.inf

    pattern_to_find = np.array([
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

    coords = corner_peaks(corner_harris(img), min_distance=5)
    coords_subpix = corner_subpix(img, coords, window_size=11)

    for coord in coords_subpix:
        x, y = coord
        if np.isnan(x) or np.isnan(y):
            continue
        x = int(round(x))
        y = int(round(y))
        dist = pattern_dist(img, x, y, pattern_to_find) 
        if dist < min_dist:
            min_dist = dist
            min_x, min_y = x, y

    return min_x, min_y

def update_marker(img, prev_x, prev_y, prev_pattern):
    min_dist = np.inf

    # the search region is the window of 'W'x'W' size 
    # centered around 'prev_center'
    W = 31

    for x in range(prev_x - W // 2, prev_x + W // 2 + 1):
        for y in range(prev_y - W // 2, prev_y + W // 2 + 1):
            if np.isnan(x) or np.isnan(y):
                continue
            x = int(round(x))
            y = int(round(y))
            dist = pattern_dist(img, x, y, prev_pattern) 
            if dist < min_dist:
                min_dist = dist
                min_x, min_y = x, y
    
    return min_x, min_y

def track_subj_marker(subj_root, visualize=False):
    print('Tracking head marker for subject: ' + subj_root)

    img_paths = ImgPathGenerator(subj_root)

    with open(os.path.join(img_paths.get_root(), '_MarkerPos.txt')) as f:
        start_x, start_y = map(int, f.readline().split(' '))

    mark_x = start_x
    mark_y = start_y
    mark_pattern = None

    # TODO: rewrite it in the way that allows to do it simultaneously
    # with all preprocessing steps
    head_mov_path = os.path.join(img_paths.get_root(), 'head_mov.txt')
    head_mov_file = open(head_mov_path, 'w')

    for img_path in img_paths:
        img = imread(img_path, as_gray=True)
        if mark_pattern is None:
            mark_pattern = init_pattern(img, start_x, start_y)
        else:
            mark_x, mark_y = update_marker(img, mark_x, mark_y, mark_pattern)
        print(start_x - mark_x, start_y - mark_y, file=head_mov_file)

        # TODO: get rid of opencv for visualization
        if visualize:
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            cv2.circle(img, (mark_y, mark_x), 3, (255,0,255), 2)
            cv2.imshow("eye", img)
            if cv2.waitKey(1) == ord('q'):
               continue
               break

def track_markers(dataset_root):
    for subj_root in os.listdir(dataset_root):
        track_subj_marker(os.path.join(dataset_root, subj_root))
        break

if __name__ == "__main__":
    track_markers(sys.argv[1])