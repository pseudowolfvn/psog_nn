import numpy as np

from sklearn.externals import joblib

from .finetune import load_and_finetune, load_and_finetune_fc
from .from_scratch import train_from_scratch
from .load_data import split_test_from_train
from .utils import get_module_prefix
from utils.utils import get_arch


def evaluation(test_subjs, params, REPS=1):
    arch = get_arch(params)
    data = {'subjs': test_subjs}
    for subj in test_subjs:
        ft = np.zeros((REPS))
        ft_time = np.zeros((REPS))
        ft_fc = np.zeros((REPS))
        ft_fc_time = np.zeros((REPS))
        scr = np.zeros((REPS))
        scr_time = np.zeros((REPS))
        
        for i in range(REPS):
            _, acc, t = load_and_finetune(test_subjs, subj, params)
            ft[i] = acc
            ft_time[i] = t
        
        data[subj] = {}
        data[subj]['ft'] = {}
        data[subj]['ft']['data'] = ft
        data[subj]['ft']['time'] = ft_time

        if arch == 'cnn':
            for i in range(REPS):
                _, acc, t = load_and_finetune_fc(test_subjs, subj, params)
                ft_fc[i] = acc
                ft_fc_time[i] = t
            data[subj]['ft_fc'] = {}
            data[subj]['ft_fc']['data'] = ft_fc
            data[subj]['ft_fc']['time'] = ft_fc_time

        for i in range(REPS):
            _, acc, t = train_from_scratch(subj, params)
            scr[i] = acc
            scr_time[i] = t
        data[subj]['scr'] = {}
        data[subj]['scr']['data'] = scr
        data[subj]['scr']['time'] = scr_time

    print(data)
    data_path = os.path.join(
        get_module_prefix(),
        str(arch) + '_' + str(params) + '_' + str(test_subjs) + '.pkl'
    )
    joblib.dump(data, data_path)

def cross_testing(test_subjs, params):
    train_subjs, test_subjs = split_test_from_train(test_subjs)
    print('Train on: ', train_subjs, 'Test on: ', test_subjs)
    train_and_save(train_subjs, test_subjs, params, load=True)
    evaluation(test_subjs, params)

if __name__ == "__main__":
    # BEST_MLP_MODEL = (0, 0, 4, 96)
    # BEST_CNN_MODEL = (1, 4, 3, 96)
    # BEST_MLP_MODEL = (0, 0, 6, 20)
    BEST_CNN_MODEL = (4, 4, 4, 20)

    # cross_testing(['1', '2', '3', '4'], BEST_MLP_MODEL)
    # cross_testing(['5', '6', '7', '8'], BEST_MLP_MODEL)
    # cross_testing(['10', '11', '12', '13'], BEST_MLP_MODEL)
    # cross_testing(['14', '15', '16', '17'], BEST_MLP_MODEL)
    # cross_testing(['18', '19', '20'], BEST_MLP_MODEL)
    # cross_testing(['21', '22', '23'], BEST_MLP_MODEL)

    cross_testing(['1', '2', '3', '4'], BEST_CNN_MODEL)
    cross_testing(['5', '6', '7', '8'], BEST_CNN_MODEL)
    cross_testing(['10', '11', '12', '13'], BEST_CNN_MODEL)
    cross_testing(['14', '15', '16', '17'], BEST_CNN_MODEL)
    cross_testing(['18', '19', '20'], BEST_CNN_MODEL)
    cross_testing(['21', '22', '23'], BEST_CNN_MODEL)