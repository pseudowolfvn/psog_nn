import os
import sys

from ml.utils import get_module_prefix
from ml.model import build_model
from utils.metrics import calc_acc
from .load_data import get_general_data, split_test_from_all
from .power_efficiency import mlp_flops, cnn_flops


def get_best_model_params(arch, setup):
    log_dir = os.path.join(get_module_prefix(), 'log')
    if not os.path.exists(log_dir):
        print('Directory with grid-search results doesn\'t exist!')
        print('Run grid-search first')
    
    for filename in os.listdir(log_dir):
        if arch in filename and filename.startswith(setup):
            log_path = os.path.join(log_dir, filename)
            log = open(log_path, 'r')

            for line in log.readlines():
                if line[0] == '(':
                    nums = line[1:-2].split(', ')
                    # for low-power setup only 
                    # log contains FLOPS complexity of the model
                    if setup == 'lp':
                        nums = nums[:-1]    
                    if arch == 'mlp':
                        params = tuple(map(int, nums[-2:]))
                        # workaround for old ETRA-like style of grid-search log
                        params = (0, 0) + params
                        return params
                    elif arch == 'cnn':
                        params = tuple(map(int, nums[-4:]))
                        return params

    print('Log file with grid-search results doesn\'t exist!')
    print('Run grid-search for architecture:', arch,
        ', setup:', setup)
    return None

def grid_search_arch_setup(root, train_subjs, test_subjs,
    search_space, arch, setup, redo):
    log_dir = os.path.join(get_module_prefix(), 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    log_path = os.path.join(log_dir, setup + '_' + arch + '_grid_search')

    if not redo and os.path.exists(log_path):
        print('Grid-search for architecture:', arch,
            ', setup:', setup, ' is already done, skip.')
        return
    
    log = open(log_path, 'w')

    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_general_data(root, train_subjs, test_subjs, arch)

    models = []

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
                        flops = mlp_flops(L_fc, N) if arch == 'mlp' \
                            else cnn_flops(L_conv, D, L_fc, N)
                        if flops > 84000:
                            continue
                    print('Fitting the model with params: ' + 
                        str([D]*L_conv) +
                        ', ' + str([N]*L_fc), file=log)
                    
                    model = build_model((L_conv, D, L_fc, N))
                    fit_time = model.train(
                        X_train, y_train, X_val, y_val,
                        batch_size=2000
                    )
                    
                    train_acc, test_acc, val_acc = model.report_acc(
                        X_train, y_train,
                        X_test, y_test,
                        X_val, y_val,
                    )
                    print('val ACC = {}, train ACC = {}, test ACC = {}, time = {}'
                        .format(val_acc, train_acc, test_acc, fit_time),  file=log)
                    models.append(
                        (val_acc, train_acc, test_acc, 
                            fit_time, 
                            L_conv, D, L_fc, N)
                    )
                    log.flush()

    models.sort()
    for model in models:
        print(model, file=log)

    log.close()
    return models[0]

def grid_search(root, archs, setups, redo=True):
    train_subjs, test_subjs = split_test_from_all([])
    # we want to ensure that generalization for unseen subjects is very poor,
    # so we will take out last subject for test purposes
    test_subjs = train_subjs[-1]
    train_subjs = train_subjs[:-1]

    search_space = {
        'mlp': {
            'lp': {
                'layers': [3,4,5,6],
                'neurons': [16,20,24,28,32]
            },
            'hp': {
                'layers': [3,4,5,6],
                'neurons': [16,32,48,64,96,128]
            },
        },
        'cnn': {
            'lp': {
                'conv_layers': [1,2,4],
                'conv_depth': [4,8,16],
                'layers': [3,4,5,6],
                'neurons': [16,20,24,28,32]
            },
            'hp': {
                'conv_layers': [1,2,4],
                'conv_depth': [4,8,16],
                'layers': [3,4,5,6],
                'neurons': [16,32,48,64,96,128]
            },
        },
    }

    for arch in archs:
        for setup in setups:
            grid_search_arch_setup(root, train_subjs, test_subjs,
                search_space[arch][setup], arch, setup, redo)

def grid_search_test_etra(root, archs, setups):
    train_subjs, test_subjs = split_test_from_all([])
    # we want to ensure that generalization for unseen subjects is very poor,
    # so we will take out last subject for test purposes
    test_subjs = train_subjs[-1]
    train_subjs = train_subjs[:-1]
    
    search_space = {
        'mlp': {
            'lp': {
                'layers': [6],
                'neurons': [20]
            },
            'hp': {
                'layers': [4],
                'neurons': [96]
            },
        },
        'cnn': {
            'lp': {
                'conv_layers': [4],
                'conv_depth': [4],
                'layers': [4],
                'neurons': [20]
            },
            'hp': {
                'conv_layers': [1],
                'conv_depth': [4],
                'layers': [3],
                'neurons': [96]
            },
        },
    }

    for arch in archs:
        for setup in setups:
            grid_search_arch_setup(root, train_subjs, test_subjs,
                search_space[arch][setup], arch, setup, redo)

if __name__ == '__main__':
    grid_search_test_etra(sys.argv[1], ['mlp', 'cnn'], ['lp', 'hp'])