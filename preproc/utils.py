import numpy as np

from utils.utils import repeat_up_to

def get_cyclic_shifts(hor, ver, N):
    """Get all possible simulated sensor shifts,
        that are repeated in a cyclic way to match the size of data.

    Args:
        hor: A list of all possible values for sensor shift horizontally.
        ver: A list of all possible values for sensor shift vertically.
        N: An int with number of shifts. 

    Returns:
        A numpy array of (N, 2) shape.
    """
    shift_pairs = np.array([(h, v) for h in hor for v in ver])
    shifts = repeat_up_to(shift_pairs, N)
    return shifts

def get_default_shifts(N):
    hor = ver = np.arange(-2., 2. + 0.1, 0.5)
    return get_cyclic_shifts(hor, ver, N)

def get_no_shifts(N):
    hor = ver = [0.] * 9
    return get_cyclic_shifts(hor, ver, N)

def get_randn_shifts(N):
    shifts = np.random.normal(size=(N, 2))
    return shifts
