import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

from .self_calibration import self_calibration
from utils.metrics import calc_acc
from utils.utils import to_radial, merge_sorted_unique, get_eye_stim_signal

def roll_with_zero_pad(data, shift, left=True):
    if left:
        return np.pad(data, (0, shift), 'constant')[shift:]
    else:
        return np.pad(data, (shift, 0), 'constant')[:-shift]

def roll_with_zero_pad_2D(data, shift, left=True):
    roll = lambda x, axis: roll_with_zero_pad(x[:,axis], shift, left)
    return zip(roll(data, 0), roll(data, 1))


def best_correlation_shift(data):
    radial_eye = np.ravel(data[['radial_eye']].values)
    radial_stim = np.ravel(data[['radial_stim']].values)

    # try all possible shift in a reasonable range
    # and find one that gives best correlation
    best_corr = 0.
    best_shift = 0
    MAX_SHIFT = 400
    for shift in range(1, MAX_SHIFT + 1):
        shifted_eye = roll_with_zero_pad(radial_eye, shift)
        corr, _ = pearsonr(shifted_eye, radial_stim)
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    print('Best correlation: {}, by {} shift'.format(best_corr, best_shift))
    # shift column in DataFrame
    data[['eye_x', 'eye_y']] = data[['eye_x', 'eye_y']].shift(-best_shift)
    # discard the last part of the signal which is NaN after shift
    data = data.dropna()

    return data

def remove_fixation_outliers(data):
    data['acc'] = data.apply(lambda x: calc_acc(np.array([[x.eye_x, x.eye_y]])
        , np.array([[x.stim_x, x.stim_y]])), axis=1)

    data.reset_index(drop=True, inplace=True)
    # use 'stim_' pattern not to catch 'radial_stim'
    targets_begin = data.ne(data.shift()).filter(like='stim_') \
       .apply(lambda x: x.index[x].tolist()).values
    targets_begin = merge_sorted_unique(*targets_begin)

    # how many points from each fixation to save
    length = 9**2 # * 5

    SKIP = 450

    # lets discard first fixation
    # data.drop(data.index[0: targets_begin[1] - 1], inplace=True)

    # TODO: end up doing append to initially empty DataFrame
    # but it's very memory inefficient
    filtered_data = pd.DataFrame(columns=data.columns)

    for t in range(1, targets_begin.shape[0] - 1):
        begin = targets_begin[t]
        end = targets_begin[t + 1] - 1
        begin = begin + SKIP
        target = data[begin: end]
        
        # TODO: decide about filtration technique
        # target.drop(
        #     target[target.acc > 0.1].index
        #         , inplace=True
        # )

        std_eye = target['radial_eye'].std()
        mean_eye = target['radial_eye'].mean()
        mean_stim = target['radial_stim'].mean()
        # remove what is away from 3 SD
        target.drop(
           target[np.abs(target.radial_eye - mean_eye) > 3*std_eye].index
               , inplace=True)
        # remove what is away from 2 degrees from eye position mean
        target.drop(
           target[np.abs(data.radial_eye - mean_eye) > 2].index
               , inplace=True)
        # remove what is away from 5 degrees from target position mean
        target.drop(
           target[np.abs(data.radial_eye - mean_stim) > 5].index
               , inplace=True)

        # HISTORICAL ARTIFACTS
        # data.drop(data.index[begin: begin + SKIP], inplace=True)
        # std_eye = data[(data.index >= begin) & (data.index <= end)]['radial_eye'].std()
        # data.drop(data[(data.index >= begin) & (data.index <= end)
        #     & (np.abs(data.radial_eye - mean_eye) > 3*std_eye)].index, inplace=True)
        
        # save only first 'length' samples from that fixation
        # if not enough then skip it
        target.reset_index(drop=True, inplace=True)
        if len(target.index) < length:
            continue
        filtered_data = filtered_data.append(target[:length], ignore_index=True)

    return filtered_data

def filtration(data):
    # calculate radial eye and stimulus coordinates
    data['radial_eye'] = data.apply(
        lambda x: to_radial( (x['eye_x'], x['eye_y']) ), axis=1) 
    data['radial_stim'] = data.apply(
        lambda x: to_radial( (x['stim_x'], x['stim_y']) ), axis=1) 

    # it's a trick to check whether shift is present
    # TODO: try to look for best shift!
    data_acc = calc_acc(*get_eye_stim_signal(data))
    data['eye_y'] += 1.2
    shifted_acc = calc_acc(*get_eye_stim_signal(data))
    if shifted_acc > data_acc:
        data['eye_y'] -= 1.2

    data = best_correlation_shift(data)
    data = remove_fixation_outliers(data)
    # data = self_calibration(data)
    return data

def discard_missing(data):
    data = data[data.valid != 4]
    return data