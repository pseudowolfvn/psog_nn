import sys

from preproc.shift_crop import shift_and_crop
from preproc.restore_missed import restore_missed_samples
from preproc.head_mov_tracker import track_markers
from preproc.psog import simulate_psog

if __name__ == '__main__':
    # TODO: merge following to a separate preprocessing function
    root = sys.argv[1]
    restore_missed_samples(root)
    track_markers(root)
    shift_and_crop(root)
    simulate_psog(root)