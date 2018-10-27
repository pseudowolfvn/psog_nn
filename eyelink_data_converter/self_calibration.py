import matplotlib.pyplot as plt

from utils.utils import merge_sorted_unique



def self_calibration(data):
    data.reset_index(drop=True, inplace=True)
    # use 'stim_' pattern not to catch 'radial_stim'
    targets_begin = data.ne(data.shift()).filter(like='stim_') \
       .apply(lambda x: x.index[x].tolist()).values
    targets_begin = merge_sorted_unique(*targets_begin)
    
    WINDOW_SIZE = 100
    WINDOW_MIN_SIZE = 10
    MADs = []
    for ind in range(0, targets_begin.shape[0] - 1):
        begin = targets_begin[ind]
        end = targets_begin[ind + 1] - 1
        for w in range(begin, end, WINDOW_SIZE):
            window = data[w: min(end, w + WINDOW_SIZE - 1)]
            # skip window if it's too small
            if len(window.index) < WINDOW_MIN_SIZE:
                continue
            MADs.append(window['eye_x'].mad())
    # plt.hist(MADs, bins=100)
    # plt.show()
    return data