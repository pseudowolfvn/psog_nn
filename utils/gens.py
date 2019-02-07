""" Images generators.
"""
import os
from pathlib import Path

import pandas as pd
from skimage.io import imread

class ImgPathGenerator:
    def __init__(self, subj_root, dirname='images'):
        self.root = os.path.join(subj_root, dirname)
        self.img_ind = 0

    def __next__(self):
        while os.path.exists(self.get_img_path()):
            img_path = self.get_img_path()
            self.img_ind += 1
            return img_path
        raise StopIteration()

    def __iter__(self):
        return self

    def get_img_path(self):
        return os.path.join(self.root, str(self.img_ind) + '.jpg')

    def get_root(self):
        return self.root

# TODO: make the following class abstract
class ImgSampleGenerator:
    def __init__(self, subj_root, imgs_dir, data_relpath):
        self.img_paths = ImgPathGenerator(subj_root, imgs_dir)
        self.data = pd.read_csv(
            os.path.join(subj_root, data_relpath),
            sep='\t'
        )

    def __next__(self):
        for i, img_path in enumerate(self.img_paths):
            # TODO: there is inconsistency between the number of headers
            # and actual number of data columns in the full signal, so
            # we need to check for all NaN except the timestamp, not any one
            if i >= self.data.shape[0] or self.data.iloc[i][1:].isna().all():
                continue
            img = imread(img_path, as_gray=True)
            sample = self.data.iloc[i]
            return img, sample
        raise StopIteration()

    def __iter__(self):
        return self

    def get_root(self):
        return self.img_paths.get_root()

    def get_data(self):
        return self.data

class FullImgSampleGenerator(ImgSampleGenerator):
    def __init__(self, subj_root):
        imgs_dir = 'images'
        data_relpath = 'FullSignal.csv'
        super().__init__(subj_root, imgs_dir, data_relpath)

class CropImgSampleGenerator(ImgSampleGenerator):
    def __init__(self, subj_root):
        imgs_dir = 'images_crop'
        data_relpath = os.path.join(imgs_dir, Path(subj_root).name + '.csv')
        super().__init__(subj_root, imgs_dir, data_relpath)
