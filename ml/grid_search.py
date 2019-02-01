import sys

from ml.utils import get_module_prefix
from ml.model import ModelMLP, ModelCNN
from utils.metrics import calc_acc
from .power_efficiency import mlp_flops, cnn_flops


def cnn_grid_search(train_subjs, test_subjs, search_space, setup):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_general_data(train_subjs, test_subjs, 'cnn')

    filename = os.path.join(
        get_module_prefix(),
        'log',
        setup + '_cnn_grid_search'
    )
    file = open(filename, 'w')
    models = []
    for L_conv in search_space['conv_layers']:
        for D in search_space['conv_depth']:
            for L_fc in search_space['layers']:
                for N in search_space['neurons']:
                    if setup == 'lp' and cnn_flops(L_conv, D, L_fc, N) > 84000:
                        continue
                    print('Fitting the model with arch: ' + 
                        str([D]*L_conv) +
                        ', ' + str([N]*L_fc), file=file)
                    
                    model = ModelCNN(L_conv, D, L_fc, N)
                    fit_time = 
                        model.train(X_train, y_train, X_val, y_val,
                            batch_size=2000)
                    
                    train_acc, test_acc, val_acc = model.report_acc(
                        X_train, y_train,
                        X_test, y_test,
                        X_val, y_val,
                    )
                    print('validation ACC = {}, train ACC = {}, test ACC = {}, time = {}'
                        .format(val_acc, train_acc, test_acc, fit_time),  file=file)
                    models.append(
                        (val_acc, train_acc, test_acc, 
                            time, 
                            L_conv, D, L_fc, N)
                    )
                    file.flush()

    models.sort()
    for model in models:
        print(model, file=file)

    file.close()
    return models[0]

def mlp_grid_search(train_subjs, test_subjs, search_space, setup):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_general_data(train_subjs, test_subjs, 'mlp')
    n_pca = X_train.shape[1]

    filename = os.path.join(
        get_module_prefix(),
        'log',
        setup + '_mlp_grid_search'
    )
    file = open(filename, 'w')
    models = []
    for L in search_space['layers']:
        for N in search_space['neurons']:
            if setup == 'lp' and mlp_flops(n_pca, L, N) > 84000:
                continue
            print('Fitting the model with arch: ' + 
                str([N]*L), file=file)
            
            model = ModelMLP(L, N)
            fit_time = 
                model.train(X_train, y_train, X_val, y_val, batch_size=2000)
            
            train_acc, test_acc, val_acc = model.report_acc(
                X_train, y_train,
                X_test, y_test,
                X_val, y_val,
            )
            print('validation ACC = {}, train ACC = {}, test ACC = {}, time = {}'
                .format(val_acc, train_acc, test_acc, fit_time),  file=file)
            models.append(
                (val_acc, train_acc, test_acc, time, L, N)
            )
            file.flush()

    models.sort()
    for model in models:
        print(model, file=file)

    file.close()
    return models[0]

def grid_search(root, archs, setups)
    if 'mlp' in archs and 'lp' in setups:
        search_space = {
            'layers': [3,4,5,6],
            'neurons': [16,20,24,28,32]
        }
        mlp_grid_search(_, _, search_space, 'lp')
    if 'cnn' in archs and 'lp' in setups:
        search_space = {
            'conv_layers': [1,2,4],
            'conv_depth': [4,8,16],
            'layers': [3,4,5,6],
            'neurons': [16,20,24,28,32]
        }
        cnn_grid_search(_, _, search_space, 'lp')
    if 'mlp' in archs and 'hp' in setups:
        search_space = {
            'layers': [3,4,5,6],
            'neurons': [16,32,48,64,96,128]
        }
        mlp_grid_search(_, _, search_space, 'hp')
    if 'cnn' in archs and 'hp' in setups:
        search_space = {
            'conv_layers': [1,2,4],
            'conv_depth': [4,8,16],
            'layers': [3,4,5,6],
            'neurons': [16,32,48,64,96,128]
        }
        cnn_grid_search(_, _, search_space, 'hp')

if __name__ == '__main__':
    grid_search(sys.argv[1], ['mlp', 'cnn'], ['lp', 'hp'])