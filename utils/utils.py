import numpy as np

def to_radial(x):
    if isinstance(x, tuple):
        return np.hypot(x[0], x[1])
    else:
        return np.hypot(x[:,0], x[:,1])

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

def repeat_up_to(arr, size):
    times = size // arr.shape[-1] + int(size % arr.shape[-1])
    return np.tile(arr, (times, 1))[:size]

def do_sufficient_pad(img, pad):
    return np.pad(img, pad, 'reflect')

def calc_pad_size(img, top_lefts, shapes):
    bottom_rights = np.array(top_lefts) + shapes
    padding = np.abs(min(
        np.min(top_lefts),
        np.min(np.array(img.shape) - bottom_rights)
    ))

    return top_lefts + padding, padding