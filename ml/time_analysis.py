""" Basic training time complexity analysis.
"""
import sys

from ml.eval import StudyEvaluation

def evaluate(root, archs, setups, redo=True):
    eval = StudyEvaluation(root, archs, setups, 'time', redo=redo)
    # smaller batch size
    eval.run({
        'batch_size': 200,
        'patience': 50
    }, 1)
    # bigger batch size
    eval.run({
        'batch_size': 2000,
        'patience': 100
    }, 1)

if __name__ == "__main__":
    evaluate(sys.argv[1], ['mlp'], ['lp'])