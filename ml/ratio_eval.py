import sys

from ml.eval import StudyEvaluation
from ml.load_data import make_get_ratio_data, get_loo_subjs_split
from ml.utils import report_results

def evaluate_study(root, archs, setups, ratio, redo=False):
    config = {'batch_size': 2000}

    eval = StudyEvaluation(root, archs, setups,
        study_id='train_' + str(ratio), redo=redo
    )
    ratio_func = make_get_ratio_data(ratio, 43)
    return eval.run(
        learning_config=config, reps=5,
        split_source=get_loo_subjs_split, data_source=ratio_func
    )

if __name__ == "__main__":
    report_results(evaluate_study(sys.argv[1], ['cnn'], ['lp'], 0.2), 'train_0.2_70_30')
    report_results(evaluate_study(sys.argv[1], ['cnn'], ['lp'], 0.4), 'train_0.4_70_30')
    report_results(evaluate_study(sys.argv[1], ['cnn'], ['lp'], 0.6), 'train_0.6_70_30')
    report_results(evaluate_study(sys.argv[1], ['cnn'], ['lp'], 0.8), 'train_0.8_70_30')
    report_results(evaluate_study(sys.argv[1], ['cnn'], ['lp'], 1.0), 'train_1.0_70_30')