""" PSOG output simulation.
"""
import os
from pathlib import Path
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from utils.gens import CropImgSampleGenerator
from utils.utils import calc_pad_size, do_sufficient_pad, find_record_dir

# this part of code is taken from Raimondas Zemblys
def gauss(w, sigma):
    """Compute gaussian kernel.

    Args:
        w: An int with kernal size.
        sigma: A float with sigma value of underlying distribution.

    Returns:
        An array with the kernel.
    """
    ax = np.arange(-w // 2 + 1., w // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    k = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = (k - np.max(k)) / np.ptp(k) + 1
    return kernel

class PSOG:
    """Class that represents PSOG sensor simulation.
    """
    def __init__(self):
        """Inits PSOG with its architecture.
        """
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
        """Get names of elements in PSOG sensor array.

        Returns:
            A list with names.
        """
        return ['psog' + str(i) for i in range(self.size)]

    def __shape_changed(self, img):
        """Check if shape of the image for which
            sensor architecture is cached didn't change.

        Args:
            img: An image of type convertible to numpy array.
        """
        return not np.equal(self.img_shape, img.shape).all()

    def __calc_sensor_layout(self, img, sensor_offset=(20, 0)):
        """Compute the PSOG sensor array layout.

        Args:
            img: An image of type convertible to numpy array.
            sensor_offset: A tuple with vertical and horizontal
                offset of layout center from the image center.
        
        Returns:
            A tuple with sensors top left coordinates, corresponding shapes.
        """
        self.img_shape = np.array(img.shape)
        img_center = self.img_shape // 2
        sensor_centers = img_center + sensor_offset + self.sensor_locations

        sensor_shapes = 2*self.sensor_sizes + 1
        sensor_top_lefts = sensor_centers - self.sensor_sizes

        return sensor_top_lefts, sensor_shapes

    def calc_layout_and_pad(self, img):
        """Compute and cache the PSOG sensor array layout,
            do sufficient image padding if some sensors
            run out of its boundaries.
        
        Args:
            img: An image of type convertible to numpy array.

        Returns:
            A tuple with image padded if needed,
                sensors top left coordinates, their corresponding shapes.
        """
        if self.__shape_changed(img):
            self.top_lefts, self.shapes = self.__calc_sensor_layout(img)
            self.top_lefts, self.pad = \
                calc_pad_size(img, self.top_lefts, self.shapes)

        img = do_sufficient_pad(img, self.pad)
        return img, self.top_lefts, self.shapes

    def simulate_output(self, img):
        """Simluate PSOG sensor output on provided image.

        Args:
            img: An image of type convertible to numpy array.

        Returns:
            An array with simulated output.
        """
        img, top_lefts, shapes = self.calc_layout_and_pad(img)

        output = np.zeros((self.size))

        for i, (tl, (h, w)) in enumerate(zip(top_lefts, shapes)):
            x, y = tl
            patch = img[x: x + h, y: y + w]
            output[i] = np.mean(patch * gauss(w, w / 4.))

        return output

    def plot_layout(self, img):
        """Visualize PSOG sensor array layout on provided image.

        Args:
            img: An image of type convertible to numpy array.
        """
        _, ax = plt.subplots(1)

        img, top_lefts, shapes = self.calc_layout_and_pad(img)
        centers = top_lefts + self.sensor_sizes
        img_plot = img.copy()

        for i, (tl, sh, c) in enumerate(zip(top_lefts, shapes, centers)):
            x, y = tl
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
    """Simluate PSOG sensor output in the whole recording for provided subject.

    Args:
        subj_root: A string with full path to directory
            with subject's recording stored in images.
    """
    print('Simulating PSOG output for subject:', subj_root)

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
        os.path.join(subj_root, data_name),
        sep='\t',
        index=False
    )


def simulate_psog(dataset_root, subj_ids=None):
    """Simluate PSOG sensor output for the provided list of subjects.

    Args:
        dataset_root: A string with path to dataset.
        subj_ids: A list with subjects ids to simulate for if provided,
            otherwise simulate for the whole dataset.
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
        simulate_subj_psog(subj_root)

if __name__ == '__main__':
    simulate_psog(sys.argv[1])
