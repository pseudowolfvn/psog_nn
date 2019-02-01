import argparse
import sys

from preproc.shift_crop import shift_and_crop
from preproc.restore_missed import restore_missed_samples
from preproc.head_mov_tracker import track_markers
from preproc.psog import simulate_psog

def build_subparsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    subparsers = parser.add_subparsers(dest='cmd')

    preproc = subparsers.add_parser('preproc')
    ml = subparsers.add_parser('ml')
    plot = subparsers.add_parser('plot')

    preproc.add_argument('--missed', nargs='*', type=str)
    preproc.add_argument('--head_mov', nargs='*', type=str)
    preproc.add_argument('--shift_crop', nargs='*', type=str)
    preproc.add_argument('--psog', nargs='*', type=str)

    ml.add_argument('--grid_search', default=False, action='store_true')
    ml.add_argument('--train', default=False, action='store_true')
    ml.add_argument('--arch', nargs='*', choices=['mlp', 'cnn'])
    ml.add_argument('--setup', nargs='*', choices=['lp', 'hp'])
    
    plot.add_argument('--boxplots')
    plot.add_argument('--error_bars', nargs='*', choices=['lp', 'hp'])
    plot.add_argument('--calib_stimuli', nargs='*')

    return parser

def none_if_empty(l):
    return None if len(l) == 0 else l

if __name__ == '__main__':
    parser = build_subparsers()

    args = parser.parse_args()
    
    root = args.root

    if args.cmd == 'preproc':
        if args.missed is not None:
            subj_ids = none_if_empty(args.missed)
            restore_missed_samples(root, subj_ids)
        if args.head_mov is not None:
            subj_ids = none_if_empty(args.head_mov)
            track_markers(root, subj_ids)
        if args.shift_crop is not None:
            subj_ids = none_if_empty(args.shift_crop)
            shift_and_crop(root, subj_ids)
        if args.psog is not None:
            subj_ids = none_if_empty(args.psog)
            simulate_psog(root, subj_ids)

    elif args.cmd == 'ml':
        if args.grid_search is not None:
            pass
        if args.train is not None:
            pass
    elif args.cmd == 'plot':
        if args.boxplots is not None:
            pass
        if args.error_bars is not None:
            pass
        if args.calib_stimuli is not None:
            pass