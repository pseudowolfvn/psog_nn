import sys

from eet_data_converter.to_blender import preprocess
from eet_data_converter.restore_missed import restore_missed_samples
from eet_data_converter.head_mov_tracker import track_markers

if __name__ == '__main__':
    # TODO: merge following to a separate preprocessing function
    root = sys.argv[1]
    restore_missed_samples(root)
    track_markers(root)
    preprocess(root)