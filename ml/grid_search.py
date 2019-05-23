""" Do grid-search for the best architecture parameters.
"""
import os
import sys

from ml.load_data import get_general_data, split_test_from_all
from ml.model import build_model
from ml.power_efficiency import mlp_flops, cnn_flops
from ml.utils import get_module_prefix


def get_best_model_params(arch, setup):
    """Extract the best model parameters from the grid-search results
        for provided neural network architecture and power consumption setup.

    Args:
        arch: A string with model architecture id.
        setup: A string with power consumption id.

    Returns:
        A tuple following the format described in ml.model.build_model().
    """
    log_dir = os.path.join(get_module_prefix(), 'log')
    if not os.path.exists(log_dir):
        print('Directory with grid-search results doesn\'t exist!')
        print('Run grid-search first')

    params = None
    for filename in os.listdir(log_dir):
        if arch in filename and filename.startswith(setup):
            log_path = os.path.join(log_dir, filename)
            log = open(log_path, 'r')

            for line in log.readlines():
                if line[0] == '(':
                    nums = line[1:-2].split(', ')
                    if arch == 'mlp':
                        params = tuple(map(int, nums[-2:]))
                        # workaround for old ETRA-like style of log
                        params = (0, 0) + params
                    elif arch == 'cnn':
                        params = tuple(map(int, nums[-4:]))
                    break

    if params is None:
        print('Log file with grid-search results doesn\'t exist!')
        print('Run grid-search for architecture:', arch, ', setup:', setup)
    return params

def grid_search_arch_setup(
        root, train_subjs, test_subjs,
        search_space, arch, setup, redo
    ):
    """Grid-search for provided neural network architecture,
        power consumption setup and search space for parameters.

    Args:
        root: A string with path to dataset.
        train_subjs: A list of subjects ids to train on.
        test_subjs: A list of subjects ids to test on.
        search_space: A dict for parameters search space
            in the following format: {
                'conv_layers': <number of convolutional layers, if any>,
                'conv_depth': <number of filters in
                    each convolutional layer, if any>,
                'layers': <number of fully-connected layers>,
                'neurons': <number of neurons in
                    each fully-connected layer>
            }.
        arch: A string with model architecture id.
        setup: A string with power consumption id.
        redo: A boolean that shows if search should be done again
            if files of results already exist.

    Returns:
        An instance of ml.model.Model() with the best parameters found.
    """
    log_dir = os.path.join(get_module_prefix(), 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_path = os.path.join(log_dir, setup + '_' + arch + '_grid_search')

    if not redo and os.path.exists(log_path):
        print(
            'Grid-search for architecture:', arch,
            ', setup:', setup, ' is already done, skip.'
        )
        # TODO: read the best model from log and return it
        return

    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_general_data(root, train_subjs, test_subjs, arch)
    n_pca = X_train.shape[1]

    models = []
    log = open(log_path, 'w')
    print('Grid-search for architecture:', arch, ', setup:', setup)

    if 'conv_layers' not in search_space:
        search_space['conv_layers'] = [0]
    if 'conv_depth' not in search_space:
        search_space['conv_depth'] = [0]

    for L_conv in search_space['conv_layers']:
        for D in search_space['conv_depth']:
            for L_fc in search_space['layers']:
                for N in search_space['neurons']:
                    if setup == 'lp':
                        flops = mlp_flops(n_pca, L_fc, N) if arch == 'mlp' \
                            else cnn_flops(L_conv, D, L_fc, N)
                        if flops > 84000:
                            continue
                    print(
                        'Fitting the model with params: '
                        + str([D]*L_conv) + ', ' + str([N]*L_fc), file=log
                    )

                    model = build_model(
                        (L_conv, D, L_fc, N),
                        n_pca, {'batch_size': 2000}
                    )
                    fit_time = model.fit(X_train, y_train, X_val, y_val)

                    train_acc, test_acc, val_acc = model.report_acc(
                        X_train, y_train,
                        X_test, y_test,
                        X_val, y_val,
                    )
                    print('val ACC = {}, train ACC = {}, test ACC = {}, time = {}'
                          .format(val_acc, train_acc, test_acc, fit_time), file=log)
                    models.append((
                        val_acc, train_acc, test_acc,
                        fit_time, L_conv, D, L_fc, N
                    ))
                    log.flush()

    models.sort()
    for model in models:
        print(model, file=log)

    log.close()
    return models[0]

def grid_search(root, archs, setups, redo=True):
    """Run grid-search to pick best parameters
        for neural network architectures.

    Args:
        root: A string with path to dataset.
        archs: A list with neural network architectures to evaluate.
        setups: A list with power consumption setups to evaluate.
        redo: A boolean that shows if search should be done again
            if files of results already exist.
    """
    train_subjs, test_subjs = split_test_from_all([])
    # we want to ensure that generalization for unseen subjects is very poor,
    # so we will take out last subject for test purposes
    test_subjs = train_subjs[-1]
    train_subjs = train_subjs[:-1]

    search_space = {
        'mlp': {
            'lp': {
                'layers': [3, 4, 5, 6],
                'neurons': [16, 20, 24, 28, 32]
            },
            'hp': {
                'layers': [3, 4, 5, 6],
                'neurons': [16, 32, 48, 64, 96]
            },
        },
        'cnn': {
            'lp': {
                'conv_layers': [1, 2, 4],
                'conv_depth': [4, 8, 16],
                'layers': [3, 4, 5],
                'neurons': [16, 20, 24, 28, 32]
            },
            'hp': {
                'conv_layers': [1, 2, 4],
                'conv_depth': [4, 8, 16],
                'layers': [3, 4, 5],
                'neurons': [16, 32, 48, 64, 96]
            },
        },
    }

    for arch in archs:
        for setup in setups:
            grid_search_arch_setup(
                root, train_subjs, test_subjs,
                search_space[arch][setup], arch, setup, redo
            )

if __name__ == '__main__':
    grid_search(sys.argv[1], ['mlp', 'cnn'], ['lp', 'hp'])
