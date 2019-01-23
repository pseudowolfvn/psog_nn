import os
import sys

import numpy as np
from skimage import data
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.io import imread

from utils.img_path_gen import ImgPathGenerator

class Marker:
    def __init__(self, img, x, y, width=11, height=9):
        self.x = x
        self.y = y

        self.w = width
        self.h = height

        self.pattern = np.array(
            img[x - self.h // 2: x + self.h // 2 + 1,
                y - self.w // 2: y + self.w // 2 + 1],
            dtype=float
        ).flatten()

    def dist(self, other):
        return np.linalg.norm(self.pattern - other.pattern, ord=2)

    def update(self, next_img):
        min_dist = np.inf

        # the search region is the window of 'W'x'W' size 
        # centered around '(self.x, self.y)'
        W = 31

        # protect the search space to not run out of image boundaries 
        for x in range(
                max(self.x - W // 2, 0),
                min(self.x + W // 2 + 1, next_img.shape[0] - self.h // 2)
            ):
            for y in range(
                    max(self.y - W // 2, 0), 
                    min(self.y + W // 2 + 1, next_img.shape[1] - self.w // 2)
                ):
                x = int(round(x))
                y = int(round(y))
                dist = self.dist(Marker(next_img, x, y))
                if dist < min_dist:
                    min_dist = dist
                    min_x, min_y = x, y
        
        self.img = next_img
        self.x = min_x
        self.y = min_y

# TODO: this doesn't work and right now initial marker position is determined
# manually, come up with a better automatic algorithm
# def find_marker(img):
#     min_dist = np.inf

#     marker_to_find = Marker(np.array([
#         [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
#         [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
#         [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
#         [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
#         [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#     ]), 4, 5)

#     coords = corner_peaks(corner_harris(img), min_distance=5)
#     coords_subpix = corner_subpix(img, coords, window_size=11)

#     for coord in coords_subpix:
#         x, y = coord
#         if np.isnan(x) or np.isnan(y):
#             continue
#         x = int(round(x))
#         y = int(round(y))
#         dist = marker_to_find.dist(Marker(img, x, y))
#         if dist < min_dist:
#             min_dist = dist
#             min_x, min_y = x, y

#     return min_x, min_y

def track_subj_marker(subj_root, visualize=False):
    print('Tracking head marker for subject: ' + subj_root)

    img_paths = ImgPathGenerator(subj_root)

    with open(os.path.join(img_paths.get_root(), '_MarkerPos.txt')) as f:
        start_x, start_y = map(int, f.readline().split(' '))

    # TODO: rewrite it in the way that allows to do it simultaneously
    # with all preprocessing steps
    head_mov_path = os.path.join(img_paths.get_root(), 'head_mov_1.txt')
    head_mov_file = open(head_mov_path, 'w')

    marker = None
    for img_path in img_paths:
        img = imread(img_path, as_gray=True)
        if marker is None:
            marker = Marker(img, start_x, start_y)
        else:
            # will serve as the marker from the previos image
            # for the next iteration
            marker.update(img)
        print(marker.x - start_x, marker.y - start_y, file=head_mov_file)

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
    for dirname in os.listdir(dataset_root):
        subj_root = os.path.join(dataset_root, subj_root)
        track_subj_marker(subj_root)

if __name__ == "__main__":
    track_markers(sys.argv[1])