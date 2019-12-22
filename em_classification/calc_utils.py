import numpy as np


def finite_diff(data):
    return np.array(np.roll(data, -1, axis=0) - data)[:-1]

def enrich_data(data, stim_data):
    timestamps = stim_data[:, 0]
    stimuli_pos = stim_data[:, -2:]

    # data = calc_velocity(data)
    data = calc_radial_velocity(data)
    data = calc_stim_pos_diff(data, stimuli_pos, timestamps)

    return data

def calc_velocity(data):
    timestamp = data.time
    dt = finite_diff(timestamp)

    pos_x = data['pos_x'].values
    pos_y = data['pos_y'].values

    dpx = finite_diff(pos_x)
    dpy = finite_diff(pos_y)

    data['dpx'] = np.concatenate(([0], dpx / dt * 1000))
    data['dpy'] = np.concatenate(([0], dpy / dt * 1000))

    return data

def calc_radial_velocity(data):
    timestamp = data.time
    dt = finite_diff(timestamp)

    pos_x = data['pos_x'].values
    pos_y = data['pos_y'].values
    data['pos_rad'] = np.sqrt(pos_x ** 2 + pos_y ** 2)
    dpx = finite_diff(pos_x)
    dpy = finite_diff(pos_y)
    drp = np.sqrt(dpx ** 2 + dpy ** 2)

    data['rv'] = np.concatenate(([0], drp / dt * 1000))

    return data

def calc_stim_pos_diff(data, stimuli_pos, timestamps):
    timestamps = [0, *timestamps]
    for i in range(len(timestamps) - 1):
        t_b = timestamps[i]
        t_e = timestamps[i + 1]
        mask = (data.time >= t_b) & (data.time < t_e)
        st_x, st_y = stimuli_pos[i]
        data.loc[mask, 'diff_x'] = data.loc[mask, 'pos_x'] - st_x
        data.loc[mask, 'diff_y'] = data.loc[mask, 'pos_y'] - st_y

    data['diff_rad'] = np.sqrt(data['diff_x'] ** 2 + data['diff_y'] ** 2)

    return data

def calc_median(data, beg, end, field, take_abs=False):
    data_slice = data[
            (data.time >= beg) & (data.time <= end)
        ][field]
    if take_abs:
        data_slice = data_slice.abs()
    return data_slice.median()