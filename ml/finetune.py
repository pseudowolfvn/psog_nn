from .load_data import get_general_data, get_specific_data
from .model import Model
from .utils import get_model_name
from utils.utils import get_arch


def train_and_save(train_subjs, test_subjs, params, load=False):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_general_data(train_subjs, test_subjs, arch=get_arch(params))

    model_name = get_model_name(test_subjs, params)
    if not load or not os.path.exists(model_name):
        model = Model(*params)
        model.train(X_train, y_train, X_val, y_val, batch_size=2000)
        model.save(model_name)
        print('Model ' + model_name + ' saved')
    else:
        print('Model ' + model_name + ' already exists')
        model = load_model(model_name)
        model.summary()
        print('Model ' + model_name + ' loaded')
    
    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

def load_and_finetune(test_subjs, subj, params):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_specific_data(test_subjs, subj, True, arch=get_arch(params))

    model = load_model(get_model_name(test_subjs, params))
    model.summary()
    print('Model loaded')

    fit_time = 
        model.train(X_train, y_train, X_val, y_val,
            epochs=1000,
            batch_size=200,
            patience=50
        )

    print('Partial fit completed')
    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

    return train_acc, test_acc, fit_time

def load_and_finetune_fc(test_subjs, subj, params):
    if get_arch(params) != 'cnn':
        print('Can be called only for CNN!')
        return

    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_specific_data(test_subjs, subj, True, arch='cnn')
    
    model = load_model(get_model_name(test_subjs, params))
    model.summary()
    for layer in model.layers:
        if layer.name.startswith('conv2d'):
            layer.trainable = False
    print('Model loaded')

    fit_time = 
        model.train(X_train, y_train, X_val, y_val,
            epochs=1000,
            batch_size=2000,
            patience=50
        )

    train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
    print('Train acc: ', train_acc)
    print('Test acc: ', test_acc)

    return train_acc, test_acc, fit_time