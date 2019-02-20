"""Provides CLI.
"""
import argparse
import os

from preproc.shift_crop import shift_and_crop
from preproc.restore_missed import restore_missed_samples
from preproc.head_mov_tracker import track_markers
from preproc.psog import simulate_psog
from ml.grid_search import grid_search
from ml.eval import evaluate_study
from plots.boxplots_per_subject import plot_boxplots
from plots.error_bars import plot_error_bars
from plots.samples_distrib import draw_samples
from utils.utils import none_if_empty, list_if_not


def add_subjs_argument(parser, arg_name, help_str):
    parser.add_argument(
        arg_name, metavar='SUBJ', nargs='*', type=str, help=help_str
    )

def add_archs_argument(parser):
    archs = ['mlp', 'cnn']
    parser.add_argument(
        '--arch', default=archs, nargs='?', choices=archs,
        help='''restrict corresponding analysis to
            either MLP or CNN architecture. If not specified, run for both.'''
    )

def add_setups_argument(parser):
    setups = ['lp', 'hp']
    parser.add_argument(
        '--setup', default=setups, nargs='?', choices=setups,
        help='''restrict corresponding analysis to
            either LP (low-power) or HP (high-power) setup.
            If not specified, run for both.'''
    )

def build_subparsers():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', default='dataset', nargs=1,
        help='''path to the dataset root directory.
            If not specified, considered to be '.\\dataset'.
            Is used for the 'preproc', 'ml' modes and
            'plot' mode with --samples_distrib'''
    )
    subparsers = parser.add_subparsers(dest='cmd')

    preproc = subparsers.add_parser('preproc')
    ml = subparsers.add_parser('ml')
    plot = subparsers.add_parser('plot')

    add_subjs_argument(
        preproc, '--missed',
        '''for every 'SUBJ' subject, fill each timestamp the signal from
            the eye-tracker was lost with NaNs.
            Run on the whole dataset, if no subjects specified'''
    )
    add_subjs_argument(
        preproc, '--head_mov',
        '''for every 'SUBJ' subject, estimate head movements by tracking
            angular marker and save shifts for each timestamp to
            'ROOT\\SUBJ\\images\\head_mov.txt' file.
            Run on the whole dataset, if no subjects specified'''
    )
    add_subjs_argument(
        preproc, '--shift_crop',
        '''for every 'SUBJ' subject, load estimated head movements and
            with artificially added sensor shifts, crop appropriate image
            region to get 'inside-headset-like' close eye captions and
            save them to 'ROOT\\SUBJ\\images_crop' directory.
            Run on the whole dataset, if no subjects specified'''
    )
    add_subjs_argument(
        preproc, '--psog',
        '''for every 'SUBJ' subject, simulate PSOG sensor outputs and
            save them together with eye-movement signal
            to 'ROOT\\SUBJ\\SUBJ_<PSOG.arch>.csv' file.
            Run on the whole dataset, if no subjects specified'''
    )

    ml.add_argument(
        '--grid_search', default=False, action='store_true',
        help='''run the grid-search to find the best architecture parameters.
            Save the log with results to '.\\ml\\log' directory'''
    )
    ml.add_argument(
        '--evaluate', default=False, action='store_true',
        help='''evaluate best architectures found during grid-search
            for 'fine-tune' and 'from scratch' approaches on the
            whole dataset'''
    )
    add_archs_argument(ml)
    add_setups_argument(ml)

    plot.add_argument(
        '--boxplots', default=False, action='store_true',
        help='''create additional boxplots of spatial accuracy for per subject
            evaluation of 'fine-tune' and 'from scratch' approaches and
            save them to '.\\plots\\boxplots' directory'''
    )
    plot.add_argument(
        '--error_bars', default=False, action='store_true',
        help='''create error bars of spatial accuracy for evaluation of
            'fine-tune' and 'from scratch' approaches and save them to
            '.\\plots\\error_bars' directory'''
    )
    plot.add_argument(
        '--samples_distrib', nargs='*',
        help='''for every 'SUBJ' subject, create image that depicts samples
            distribution on the screen with attempt to resemble calibration
            data split and save them to '.\\plots\\samples_distrib' directory.
            Run on the whole dataset, if no subjects specified'''
    )
    add_archs_argument(plot)
    add_setups_argument(plot)

    return parser

def run_cli():
    parser = build_subparsers()

    args = parser.parse_args()

    dataset_root = args.root[0] if isinstance(args.root, list) else args.root
    results_root = os.path.join('ml', 'results')

    # if --arch or --setup is used with only one argument, convert it to list
    if 'arch' in args:
        args.arch = list_if_not(args.arch)
    if 'setup' in args:
        args.setup = list_if_not(args.setup)

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
        if args.evaluate:
            evaluate_study(dataset_root, args.arch, args.setup, redo=False)
    elif args.cmd == 'plot':
        if args.boxplots:
            plot_boxplots(results_root, args.arch, args.setup)
        if args.error_bars:
            plot_error_bars(results_root, args.setup)
        if args.samples_distrib is not None:
            subj_ids = none_if_empty(args.samples_distrib)
            draw_samples(dataset_root, subj_ids)

if __name__ == '__main__':
    run_cli()
