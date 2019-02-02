import argparse
import os
import sys

from preproc.shift_crop import shift_and_crop
from preproc.restore_missed import restore_missed_samples
from preproc.head_mov_tracker import track_markers
from preproc.psog import simulate_psog
from ml.grid_search import grid_search
from ml.eval import evaluate_study
from plots.boxplots_per_subject import plot_boxplots
from plots.error_bars import plot_error_bars
from plots.samples_distrib import draw_samples

def build_subparsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='dataset', nargs='?')
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
    archs = ['mlp', 'cnn']
    ml.add_argument('--arch', default=archs, nargs='*', choices=archs)
    setups = ['lp', 'hp']
    ml.add_argument('--setup', default=setups, nargs='*', choices=setups)
    
    plot.add_argument('--boxplots', default=False, action='store_true')
    plot.add_argument('--error_bars', default=False, action='store_true')
    plot.add_argument('--arch', default=archs, nargs='*', choices=archs)
    plot.add_argument('--setup', default=setups, nargs='*', choices=setups)
    plot.add_argument('--samples_distrib', nargs='*')

    return parser

def none_if_empty(l):
    return None if len(l) == 0 else l

if __name__ == '__main__':
    parser = build_subparsers()

    args = parser.parse_args()
    
    dataset_root = args.root
    results_root = os.path.join('ml', 'results')

    if args.cmd == 'preproc':
        if args.missed is not None:
            subj_ids = none_if_empty(args.missed)
            restore_missed_samples(dataset_root, subj_ids)
        if args.head_mov is not None:
            subj_ids = none_if_empty(args.head_mov)
            track_markers(dataset_root, subj_ids)
        if args.shift_crop is not None:
            subj_ids = none_if_empty(args.shift_crop)
            shift_and_crop(dataset_root, subj_ids)
        if args.psog is not None:
            subj_ids = none_if_empty(args.psog)
            simulate_psog(dataset_root, subj_ids)
    elif args.cmd == 'ml':
        if args.grid_search:
            grid_search(dataset_root, args.arch, args.setup, redo=False)
        if args.train:
            evaluate_study(dataset_root, args.arch, args.setup, redo=False)
    elif args.cmd == 'plot':
        if args.boxplots:
            plot_boxplots(results_root, args.arch, args.setup)
        if args.error_bars:
            plot_error_bars(results_root, args.setup)
        if args.samples_distrib is not None:
            subj_ids = none_if_empty(args.samples_distrib)
            draw_samples(dataset_root, subj_ids)