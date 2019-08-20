import datetime
import os

import numpy as np
import pandas as pd

from ml.from_scratch import train_from_scratch


def run_subj_comp(root, subj, batch_size, REPS, impl='torch'):
    params = (2,4,4,20)
    config = {'epochs':500, 'batch_size':batch_size, 'patience': 50}

    accs = []
    times = []
    for i in range(REPS):
        config['seed'] = i
        _, acc, fit_time = train_from_scratch(
            root, subj, params, learning_config=config, impl=impl
        )
        accs.append(acc)
        times.append(fit_time)

    return np.array(accs), np.array(times)

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

def run_frameworks_comp(root, subj_ids, frameworks, batch_size, REPS, report_name=None):
    print(
        "Comparing ML frameworks: ", frameworks,
        ", for batch size: ", batch_size,
        ", on subjects: ", subj_ids, " ...", sep=''
    )
    stats = {}
    report_cols = []
    for impl in frameworks:
        stats[impl] = {}
        report_cols.extend([
            impl + ', acc: mean',
            impl + ', acc: std',
            impl + ', time: mean',
            impl + ', acc: std'
        ]) 

    for i, subj_id in enumerate(subj_ids):
        for impl in frameworks:
            acc, fit_time = run_subj_comp(root, subj_id, batch_size, REPS, impl)
            stats[impl][i] = calc_stats(acc, fit_time)

    results = []
    def unpack_stats(stats):
        return (
            stats['acc']['mean'],
            stats['acc']['std'],
            stats['time']['mean'],
            stats['time']['std']
        )

    for i, subj_id in enumerate(subj_ids):
        print('Subject', subj_id)
        subj_results = []
        for impl in frameworks:
            impl_results = unpack_stats(stats[impl][i])
            subj_results.extend(impl_results)
            print(impl, impl_results)

        results.append(subj_results)

    report = pd.DataFrame(results, columns=report_cols)

    report_dir = 'tmp'
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    if report_name is None:
        report_name = datetime.datetime.now().isoformat().replace(':', '-')

    report.to_csv(os.path.join(report_dir, report_name + '.csv'), index=False)

def compare_frameworks(dataset_root, frameworks, batch_sizes, subj_ids=None, REPS=5):
    if subj_ids is None:
        subj_ids = []
        for dirname in os.listdir(dataset_root):
            subj_ids.append(dirname)

    for bs in batch_sizes:
        report_name = 'compare_frameworks_bs_' + str(bs)
        run_frameworks_comp(
            dataset_root, subj_ids,
            frameworks, bs, REPS, report_name
        )