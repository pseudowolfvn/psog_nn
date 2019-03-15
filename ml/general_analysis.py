""" General evaluation of approaches.
"""
import sys

from ml.eval import StudyEvaluation

def evaluate(root, archs, setups, redo=True):
    eval = StudyEvaluation(root, archs, setups, redo=redo)
    eval.run()

if __name__ == "__main__":
    evaluate(sys.argv[1], ['mlp'], ['lp'])