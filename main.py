"""Provides CLI.
"""
import argparse
import os

from preproc.shift_crop import shift_and_crop
from preproc.restore_missed import restore_missed_samples
from preproc.head_mov_tracker import track_markers
from preproc.psog import simulate_psog
from ml.general_analysis import evaluate_study
from ml.grid_search import grid_search
from ml.time_analysis import evaluate_time
from ml.calib_analysis import evaluate_calib
# from plots.boxplots_per_subject import plot_boxplots
# from plots.error_bars import plot_error_bars
# from plots.samples_distrib import draw_samples
from utils.utils import none_if_empty, list_if_not


def add_subjs_argument(subparser, arg_name, help_str):
    """Add command-line argument to provided subparser for
    restricting corresponding analysis to the set of subjects.

    Args:
        subparser: A subparser obtained from ArgumentParser.add_subparser().
        arg_name: A name for the argument that relates to the type of analysis.
        help_str: A help string that will be shown in the help page.
    """
    subparser.add_argument(
        arg_name, metavar='SUBJ', nargs='*', type=str, help=help_str
    )

def add_archs_argument(subparser):
    """Add command-line argument to provided subparser for
    choosing neural network architecture.

    Args:
        subparser: A subparser obtained from ArgumentParser.add_subparser().
    """
    archs = ['mlp', 'cnn']
    subparser.add_argument(
        '--arch', default=archs, nargs='?', choices=archs,
        help='''restrict corresponding analysis to
            either MLP or CNN architecture. If not specified, run for both.'''
    )

def add_setups_argument(subparser):
    """Add command-line argument to provided subparser for
    choosing power consumption setup.

    Args:
        subparser: A subparser obtained from ArgumentParser.add_subparser().
    """
    setups = ['lp', 'hp']
    subparser.add_argument(
        '--setup', default=setups, nargs='?', choices=setups,
        help='''restrict corresponding analysis to
            either LP (low-power) or HP (high-power) setup.
            If not specified, run for both.'''
    )

def build_subparsers():
    """Build subparsers for command-line arguments of project's modules.

    Returns:
        Parser of argparse.ArgumentParser type that contains
        all corresponding subparsers.
    """
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
    ml.add_argument(
        '--time', default=False, action='store_true',
        help='''basic training time complexity analysis for
            'fine-tune' and 'from scratch' approaches for CNN architecture
            of LP setup on the whole dataset'''
    )
    ml.add_argument(
        '--calib_like_train', default=False, action='store_true',
        help='''basic analysis of using calibration-like distribution
            of training set for 'fine-tune' approach for CNN architecture
            of both setups for subjects "6" and "8"'''
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
    """Run main command-line interface.
    """
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
        if args.time:
            evaluate_time(dataset_root, ['cnn'], ['lp'], redo=False)
        if args.calib_like_train:
            evaluate_calib(dataset_root, ['6', '8'])
    elif args.cmd == 'plot':
        if args.boxplots:
            plot_boxplots(results_root, args.arch, args.setup)
        if args.error_bars:
            plot_error_bars(results_root, args.setup)
        if args.samples_distrib is not None:
            subj_ids = none_if_empty(args.samples_distrib)
            draw_samples(dataset_root, subj_ids)

from ml.from_scratch import train_from_scratch
import numpy as np

def get_acc_time(root, subj, params, config, impl):
    accs = []
    times = []
    for i in range(5):
        config['seed'] = i
        _, acc, fit_time = train_from_scratch(
            root, subj, params, learning_config=config, impl=impl
        )
        accs.append(acc)
        times.append(fit_time)

    return np.array(accs), np.array(times)

def run_subj_comp(subj):
    root = r'D:\DmytroKatrychuk\dev\research\dataset\psog_nn\dataset'
    params = (2,4,4,20)
    config = {'epochs':500, 'batch_size':1000, 'patience': 50}

    torch_acc, torch_time = get_acc_time(root, subj, params, config, 'torch')
    keras_acc, keras_time = get_acc_time(root, subj, params, config, 'keras')

    return keras_acc, keras_time, torch_acc, torch_time

def calc_stats(acc, time):
    return {
        'acc': {
            'mean': np.mean(acc),
            'std': np.std(acc)
        },
        'time': {
            'mean': np.mean(time),
            'std': np.std(time)
        }
    }

def report_stats(stats):
    print(
        'Accuracy, mean: {:.2f}, std: {:.2f}'.format(
            stats['acc']['mean'],
            stats['acc']['std']
        )
    )
    print(
        'Time, mean: {:.2f}, std: {:.2f}'.format(
            stats['time']['mean'],
            stats['time']['std']
        )
    )

def run_keras_torch_comp():
    keras_stats = {}
    torch_stats = {}

    for i in range(1, 24):
        keras_acc, keras_time, torch_acc, torch_time = run_subj_comp(str(i))
        keras_stats[i] = calc_stats(keras_acc, keras_time)
        torch_stats[i] = calc_stats(torch_acc, torch_time)

    for i in range(1, 24):
        print('Subject', i)

        print('KERAS')
        report_stats(keras_stats[i])
        
        print('TORCH')
        report_stats(torch_stats[i])


if __name__ == '__main__':
    # run_cli()
    run_keras_torch_comp()