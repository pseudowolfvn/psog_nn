import os
import sys

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.externals import joblib

from ml.eval import StudyEvaluation
from ml.load_data import get_specific_data, get_stimuli_pos, default_source_if_none, get_calib_like_data, get_loo_subjs_split
from ml.model import build_model
from ml.utils import default_config_if_none, get_module_prefix, get_model_path
from utils.utils import deg_to_pix, pix_to_deg, get_arch

def get_grid_regions():
    make_regions = lambda split: \
        [(split[i], split[i + 1]) for i in range(0, len(split) - 1)]

    START_HOR = 0
    END_HOR = 1280
    hor_split = [START_HOR, 340, 541, 737.5, 937, END_HOR]
    hor_regions = make_regions(hor_split)

    START_VER = 0
    END_VER = 1024
    ver_split = [START_VER, 312.5, 446, 576.5, 708.5, END_VER]
    ver_regions = make_regions(ver_split)

    regions = [((h[0], v[0]), (h[1], v[1])) for h in hor_regions for v in ver_regions]
    regions_in_deg = [
        (pix_to_deg(rect[0]), pix_to_deg(rect[1])) for rect in regions
    ]

    return regions_in_deg

def print_regions(regions):
    for rect in regions:
        tl, br = rect
        print('({:.2f}, {:.2f}); ({:.2f}, {:.2f})'.format(tl[0], tl[1], br[0], br[1]))

def bounds_from_region(reg):
    tl, br = reg
    l_h = min(tl[0], br[0])
    r_h = max(tl[0], br[0])
    l_v = min(tl[1], br[1])
    r_v = max(tl[1], br[1])
    return l_h, r_h, l_v, r_v

def get_grid_regions_data(root, subj, data_source=None):
    # data_source = default_source_if_none(data_source)
    data_source = get_calib_like_data

    X_train, X_val, X_test, y_train, y_val, y_test \
        = data_source(root, subj, 'cnn')

    test_set_by_regions = {}
    regions = get_grid_regions()

    for reg in regions:
        l_h, r_h, l_v, r_v = bounds_from_region(reg)

        reg_ind = np.where(
            (y_test[:, 0] >= l_h) &
            (y_test[:, 0] <= r_h) &
            (y_test[:, 1] >= l_v) &
            (y_test[:, 1] <= r_v)
        )
        test_set_by_regions[reg] = (X_test[reg_ind], y_test[reg_ind])

    return X_train, X_val, y_train, y_val, test_set_by_regions

def load_and_finetune(root, train_subjs, subj, params, learning_config=None, data_source=None):
    learning_config = default_config_if_none(learning_config)

    arch = get_arch(params)

    X_train, X_val, y_train, y_val, test_set_by_regions = \
        get_grid_regions_data(root, subj, data_source=data_source)

    model = build_model(params)
    model_path = get_model_path(train_subjs, params)
    model.load_weights(model_path)

    if arch == 'cnn':
        model.freeze_conv()

    print('Model ' + model_path + ' loaded')

    fit_time = model.train(
        X_train, y_train, X_val, y_val,
        **learning_config
    )

    spatial_map = {}
    for k, v in test_set_by_regions.items():
        X_test, y_test = v
        if y_test.shape[0] == 0:
            spatial_map[k] = 0.
        else:
            train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
            print(train_acc, test_acc)
            spatial_map[k] = test_acc

    return spatial_map

def train_from_scratch(root, subj, params, learning_config=None, data_source=None):
    learning_config = default_config_if_none(learning_config)

    X_train, X_val, y_train, y_val, test_set_by_regions = \
        get_grid_regions_data(root, subj, data_source=data_source)

    model = build_model(params)
    fit_time = model.train(
        X_train, y_train, X_val, y_val,
        **learning_config
    )

    spatial_map = {}
    for k, v in test_set_by_regions.items():
        X_test, y_test = v
        if y_test.shape[0] == 0:
            spatial_map[k] = 0.
        else:
            train_acc, test_acc, _ = model.report_acc(X_train, y_train, X_test, y_test)
            print(train_acc, test_acc)
            spatial_map[k] = test_acc

    return spatial_map

def plot_subj_spatial_heatmap(mean_data, std_data, plot_path):
    data = np.round(mean_data, 2)[::-1]
    print(data)
    formater = np.vectorize(lambda m, s: '{:.2f} ± {:.2f}'.format(m, s))
    z_text = np.array(
        [formater(m, s) for m, s in zip(mean_data[::-1], std_data[::-1])]
    ).flatten()

    x = [-20.51, -9.94, -3.31, 3.26, 9.85, 20.51]
    y = [-16.70, -6.57, -2.16, 2.21, 6.67, 16.70]

    x_tickpos = np.arange(-0.5, 4.51, 1)
    y_tickpos = x_tickpos

    # grid_tickpos = [(h + 0.5, v + 0.5) for v in y_tickpos[:-1] for h in x_tickpos[:-1]]
    # annos = [go.layout.Annotation(x=x, y=y, text=z, showarrow=False) for (x, y), z in zip(grid_tickpos, z_text)]
    # annos_debug = [(x, y, z) for (x, y), z in zip(grid_tickpos, z_text)]

    layout = go.Layout(xaxis={
        # 'title': {'text': 'Horizontal'},
        'side': 'bottom',
        'showticklabels': True,
        'tickmode': 'array',
        'ticktext': x,
        'tickvals': x_tickpos,
        'ticks': 'outside',
    },
    yaxis={
        # 'title': {'text': 'Vertical'},
        'showticklabels': True,
        'tickmode': 'array',
        'ticktext': y,
        'tickvals': y_tickpos,
        'ticks': 'outside',
    },
    # annotations=annos
    )
    fig = ff.create_annotated_heatmap(z=data, annotation_text=z_text.reshape(5, 5), showscale=True)
    fig.update_layout(layout)
    # fig = go.Figure(data=[go.Heatmap(z=data, showscale=True)], layout=layout)
    

    fig.write_image(plot_path)

class SpatialMapsEvaluation(StudyEvaluation):
    def __init__(self, root, redo):
        super().__init__(root, ['cnn'], ['lp'], study_id='spatial_maps_calib_like', redo=redo)

    def _evaluate_approaches(
        self, train_subjs, test_subjs,
        params, setup, config, REPS, redo,
        data_source=None
    ):
        # placeholder for best model to use
        params = (2, 4, 4, 20)

        config = {
            'batch_size': 2000,
            'epochs': 500,
            'patience': 50,
        }

        results_dir = os.path.join(get_module_prefix(), 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        name_prefix = self._get_study_prefix(config)
        data_name = name_prefix + str(setup) + '_' + str(params) + '_' + \
            str(test_subjs) + '.pkl'
        data_path = os.path.join(results_dir, data_name)

        if not redo and os.path.exists(data_path):
            print('Evaluation results', data_path, 'already exists, skip')
            data = joblib.load(data_path)
            return data

        data = {'subjs': test_subjs}
        for subj in test_subjs:
            ft = np.zeros((REPS, 5, 5))
            scr = np.zeros((REPS, 5, 5))

            for i in range(REPS):
                spatial_map = load_and_finetune(
                    self.root, train_subjs, subj,
                    params, config, data_source=data_source
                )
                ft[i] = np.fromiter(spatial_map.values(), np.double).reshape(5, 5).T

            data[subj] = {}
            data[subj]['ft'] = {}
            data[subj]['ft']['mean'] = np.mean(ft, axis=0)
            data[subj]['ft']['std'] = np.std(ft, axis=0)

            for i in range(REPS):
                spatial_map = train_from_scratch(
                    self.root, subj,
                    params, config, data_source=data_source
                )
                scr[i] = np.fromiter(spatial_map.values(), np.double).reshape(5, 5).T
                for k, v in spatial_map.items():
                    tl, br = k
                    print('{:.1f}, {:.1f}; {:.1f}, {:.1f} - {:.3f}'.format(tl[0], tl[1], br[0], br[1], v))
                print(np.round(scr[i], 3))

            data[subj]['scr'] = {}
            data[subj]['scr']['mean'] = np.mean(scr, axis=0)
            data[subj]['scr']['std'] = np.std(scr, axis=0)

        print(data)

        joblib.dump(data, data_path)

        return data

    def run(self, learning_config=None, reps=5):
        subjs_split = get_loo_subjs_split
        # subjs_split = lambda : [['1']]

        return super().run(learning_config, reps, subjs_split)

def plot_spatial_heatmap(data, appr, study_id=''):
    output_dir = os.path.join('tmp', 'heatmaps' + study_id)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_subjs = np.zeros((len(data['subjs']), 5, 5))
    for i, subj in enumerate(data['subjs']):
        subj_out_dir = os.path.join(output_dir, str(subj))
        if not os.path.exists(subj_out_dir):
            os.mkdir(subj_out_dir)

        appr_map = data[subj][appr]['mean']
        appr_map_std = data[subj][appr]['std']
        
        img_path = os.path.join(subj_out_dir, str(subj) + '_' + appr + '_map.png')
        mean_path = os.path.join(subj_out_dir, str(subj) + '_' + appr + '_mean.csv')
        std_path = os.path.join(subj_out_dir, str(subj) + '_' + appr + '_std.csv')

        plot_subj_spatial_heatmap(appr_map, appr_map_std, img_path)

        pd.DataFrame(appr_map).to_csv(mean_path)
        pd.DataFrame(appr_map_std).to_csv(std_path)

        all_subjs[i] = appr_map

    all_img_path = os.path.join(output_dir, 'all_' + appr + '_map.png')
    all_mean_path = os.path.join(output_dir, 'all_' + appr + '_mean.csv')
    all_std_path = os.path.join(output_dir, 'all_' + appr + '_std.csv')

    all_subjs_mean = np.mean(all_subjs, axis=0)
    all_subjs_std = np.std(all_subjs, axis=0)

    plot_subj_spatial_heatmap(all_subjs_mean, all_subjs_std, all_img_path)

    pd.DataFrame(all_subjs_mean).to_csv(all_mean_path)
    pd.DataFrame(all_subjs_std).to_csv(all_std_path)

if __name__ == '__main__':
    eval = SpatialMapsEvaluation(sys.argv[1], False)
    data = eval.run(reps=1)
    plot_spatial_heatmap(data, 'ft', '_randn_split')
    plot_spatial_heatmap(data, 'scr', '_randn_split')
