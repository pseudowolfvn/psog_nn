""" Utility functions.
"""
import numpy as np

def to_radial(x):
    if isinstance(x, tuple):
        return np.hypot(x[0], x[1])
    return np.hypot(x[:, 0], x[:, 1])

def merge_sorted_unique(a, b):
    a = np.array(a, dtype=np.int)
    b = np.array(b, dtype=np.int)
    # print('len(a): ', len(a))
    # print('len(b): ', len(b))
    c = np.zeros((len(a) + len(b)), dtype=np.int)
    p_a = p_b = p_c = 0
    while p_a < len(a) and p_b < len(b):
        if a[p_a] < b[p_b]:
            c[p_c] = a[p_a]
            p_a += 1
        else:
            c[p_c] = b[p_b]
            p_b += 1
        while p_a < len(a) and a[p_a] == c[p_c]:
            p_a += 1
        while p_b < len(b) and b[p_b] == c[p_c]:
            p_b += 1
        p_c += 1

    if p_a < len(a):
        p_last = p_a
        last = a
    else:
        p_last = p_b
        last = b

    while p_last < len(last):
        # print('p_last: ', p_last)
        c[p_c] = last[p_last]
        while p_last < len(last) and last[p_last] == c[p_c]:
            p_last += 1
        p_c += 1

    return c[:p_c]

def get_eye_stim_signal(data):
    eye = data[['eye_x', 'eye_y']].values
    stim = data[['stim_x', 'stim_y']].values

    return eye, stim

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