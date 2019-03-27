""" General evaluation of approaches.
"""
import sys

from ml.eval import StudyEvaluation


def evaluate_study(root, archs, setups, redo=True):
    """Run the general study evaluation for the whole dataset.

    Args:
        root: A string with path to dataset.
        archs: A list with neural network architectures to evaluate.
        setups: A list with power consumption setups to evaluate.
        redo: A boolean that shows if evaluation should be done again
                if files of results already exist.
    """
    eval = StudyEvaluation(root, archs, setups, redo=redo)
    eval.run()

if __name__ == "__main__":
    evaluate_study(sys.argv[1], ['mlp'], ['lp'])