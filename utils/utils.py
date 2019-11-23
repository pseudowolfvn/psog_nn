""" Utility functions.
"""
import os

import numpy as np


def deg_to_pix(deg):
    """Convert eye gaze in degrees of visual angle to pixels on the screen.

    Args:
        deg: A tuple with horizontal, vertical component
            of the eye gaze in degrees.

    Returns:
        A tuple with converted eye gaze in pixels.
    """
    posx, posy = deg
    dist_mm = 500.
    w_mm = 374.
    h_mm = 300.
    w_pix = 1280
    h_pix = 1024
    conv = lambda data, pix, mm, dist: \
        int(round(np.tan(data / 180. * np.pi) * dist * pix/mm + pix/2.))
    return conv(posx, w_pix, w_mm, dist_mm), \
        conv(-posy, h_pix, h_mm, dist_mm)

def get_arch(params):
    """Get neural network architecture id from its parameters.

    Args:
        params: A tuple with neural network paramters following the format
            described in ml.model.build_model().

    Returns:
        A string with architecture id.
    """
    if len(params) == 4 and params[0] > 0:
        return 'cnn'
    elif len(params) == 2 or params[0] == 0:
        return 'mlp'

def repeat_up_to(arr, size):
    """Repeat array in a cyclic way such that
        last dimension will become equal to provided size.

    Args:
        arr: A numpy array to repeat.
        size: An int with expected size.

    Returns:
        A numpy array produced in aforementioned way.
    """
    times = size // arr.shape[-1] + int(size % arr.shape[-1])
    return np.tile(arr, (times, 1))[:size]

def do_sufficient_pad(img, pad_size):
    """Pad provided image using mirror-padding.

    Args:
        img: An image of type convertible to numpy array.
        pad_size: An int with pad size for both width and height.

    Returns:
        A padded image.
    """
    # if image is RGB then color channel shouldn't be padded
    pad = ((pad_size,), (pad_size,), (0,)) if len(img.shape) == 3 else pad_size
    return np.pad(img, pad, 'reflect')

def calc_pad_size(img, top_lefts, shapes):
    """Compute sufficient pad size for the image such that provided
        PSOG sensor array layout won't run out of its border.

    Args:
        img: An image of type convertible to numpy array.
        top_lefts: A list of tuples, each with sensor's top left coordinate.
        shapes: A list of tuples, each with sensor's shape.

    Returns:
        A tuple with adjusted top lefts coordinates of sensors,
            computed pad size.
    """
    top_lefts = np.array(top_lefts)
    bottom_rights = top_lefts + shapes
    overrun = min(
        np.min(top_lefts),
        np.min(np.array(img.shape)[:2] - bottom_rights)
    )
    padding = 0 if overrun > 0 else -overrun

    return top_lefts + padding, padding

def none_if_empty(it):
    """
    Args:
        it: An iterable.

    Returns:
        None if 'it' is empty, 'it' otherwise.
    """
    return None if not it else it

def list_if_not(x):
    """
    Args:
        x: Any object.

    Returns:
        'x' enclosed in the list if it's not the list, 'x' otherwise.
    """
    return [x] if not isinstance(x, list) else x

def find_filename(root, default_name, beg=None, end=None):
    data_name = default_name
    for filename in os.listdir(root):
        if (beg is None or filename.startswith(beg)) \
                and (end is None or filename.endswith(end)):
            data_name = filename
    return data_name

def extract_subj_id_from_dir(subj_dir):
    return subj_dir.split('_')[-1]

def find_record_dir(root, id):
    record_dir = ''
    for dirname in os.listdir(root):
        if not os.path.isdir(os.path.join(root, dirname)):
            continue
        if dirname.startswith('Record') \
                and extract_subj_id_from_dir(dirname) == id:
            record_dir = dirname
    return record_dir