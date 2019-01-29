import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from skimage.io import imread, imsave

from utils.gens import CropImgSampleGenerator

# this part of code is taken from Raimondas Zemblys
def gauss(w, sigma):
    ax = np.arange(-w // 2 + 1., w // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    k = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = (k - np.max(k)) / np.ptp(k) + 1
    return kernel

class PSOG:
    def __init__(self):
        # TODO: move all hard-coded stuff to a separate config file
        self.size = 3*5
        self.arch = 'grid15'
        self.sensor_sizes = np.repeat([[60, 60]], self.size, axis=0)
        self.sensor_locations = np.array([
            (h, v) for h in range(-60, 60 + 1, 60)
                for v in range(-120, 120 + 1, 60)
        ])
        self.img_shape = None
    
    def get_names(self):
        return ['psog' + str(i) for i in range(self.size)]

    def __shape_changed(self, img):
        return not np.equal(self.img_shape, img.shape).all()

    def __calc_sensor_layout(self, img, sensor_offset=[20,0]):
        self.img_shape = np.array(img.shape)
        img_center = self.img_shape // 2
        sensor_centers = img_center + sensor_offset + self.sensor_locations
        
        sensor_shapes = 2*self.sensor_sizes + 1
        sensor_bottom_lefts = sensor_centers - self.sensor_sizes

        return sensor_bottom_lefts, sensor_shapes

    def __calc_pad(self, img, bottom_lefts, shapes):
        top_rights = bottom_lefts + shapes
        padding = np.abs(min(
            np.min(bottom_lefts),
            np.min(np.array(self.img_shape) - top_rights)
        ))
        
        return padding

    def calc_layout_and_pad(self, img):
        if self.__shape_changed(img):
            self.bottom_lefts, self.shapes = self.__calc_sensor_layout(img)
            self.pad = self.__calc_pad(img, self.bottom_lefts, self.shapes)
            self.bottom_lefts += self.pad 

        img = np.pad(img, self.pad, 'reflect')
        return img, self.bottom_lefts, self.shapes

    def simulate_output(self, img):
        img, bottom_lefts, shapes = self.calc_layout_and_pad(img)

        output = np.zeros((self.size))

        for i, (bl, (h, w)) in enumerate(zip(bottom_lefts, shapes)):
            x, y = bl
            patch = img[x: x + h, y: y + w]
            output[i] = np.mean(patch * gauss(w, w / 4.))

        return output

    def plot_layout(self, img):
        _, ax = plt.subplots(1)

        img, bottom_lefts, shapes = self.calc_layout_and_pad(img)
        centers = bottom_lefts + self.sensor_sizes
        img_plot = img.copy()

        for i, (bl, sh, c) in enumerate(zip(bottom_lefts, shapes, centers)):
            x, y = bl
            h, w = sh
            patch = img[x: x + h, y: y + w]
            out = patch * gauss(w, w / 4.)
            img_plot[x: x + h, y: y + w] += out
            
            c_x, c_y = c
            ax.add_patch(patches.Circle((c_y, c_x), w / 4, fill=False))
            ax.text(c_y, c_x, s=i, ha='center', va='center', color='red')

        ax.imshow(img_plot, cmap='gray')
        plt.show()

def simulate_subj_psog(subj_root):
    img_samples = CropImgSampleGenerator(subj_root)
    psog = PSOG()

    # psog_outputs = np.array([[psog.simulate_output(img)] for img, _ in img_samples])

    psog_outputs = np.zeros((img_samples.get_data().shape[0], psog.size))

    for i, (img, _) in enumerate(img_samples):    
        psog_outputs[i] = psog.simulate_output(img)

    data_name = Path(subj_root).name + '_' + psog.arch + '.csv'
    img_samples.get_data().assign(
        **dict(
            zip(psog.get_names(), psog_outputs.T)
        )
    ).to_csv(
        os.path.join(img_samples.get_root(), data_name),
        sep='\t'
    )


def simulate_psog(dataset_root):
    for dirname in os.listdir(dataset_root):
        subj_root = os.path.join(dataset_root, dirname)
        simulate_subj_psog(subj_root)
        break

if __name__ == '__main__':
    simulate_psog(sys.argv[1])