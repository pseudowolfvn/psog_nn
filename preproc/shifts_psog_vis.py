import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from skimage.io import imread, imsave

from ml.load_data import get_subj_data, normalize
from preproc.psog import PSOG
from preproc.shift_crop import get_shifted_crop
from utils.gens import ImgPathGenerator


def psog_plot(y, plot_path):
    y_labels = list(map(lambda z: '{:.3f}'.format(z), y))

    fig = go.Figure(data=[go.Bar(
        x=np.arange(0, len(y)), y=y, text=y_labels, textposition='outside',
        outsidetextfont={'size':18}, constraintext='none', textangle=90
    )])

    fig.update_layout(
        yaxis=dict(
            range=[0., 0.4]
        )
    )
    fig.write_image(plot_path)

def calc_normalized_psog_diff(psog, base_img, shifted_img):
    base_output = psog.simulate_output(base_img)
    shifted_output = psog.simulate_output(shifted_img)

    return np.abs(base_output - shifted_output) / base_output

def plot_shifts(etra_data, randn_data, plot_path):
    etra_color = 'rgb(252, 103, 3)'
    randn_color = 'rgb(3, 90, 252)'
    data=[
        go.Scatter(x=etra_data[:, 0], y=etra_data[:, 1], mode='markers', marker={'size': 10., 'color': etra_color}),
        go.Scatter(x=randn_data[:, 0], y=randn_data[:, 1], mode='markers', marker={'size': 1.5, 'color': randn_color}),
        go.Histogram(x=randn_data[:, 0], yaxis='y2', marker_color=randn_color),
        go.Histogram(y=randn_data[:, 1], xaxis='x2', marker_color=randn_color),
    ]

    AREA = 0.80
    layout = go.Layout(
        showlegend=False,
        # autosize=True,
        width=1000,
        height=1000,
        xaxis=dict(
            domain=[0, AREA],
            showgrid=False,
            range=[-3, 3],
            tickfont={'size': 18.},
            ticks='outside',
        ),
        yaxis=dict(
            domain=[0, AREA],
            showgrid=False,
            range=[-3, 3],
            tickfont={'size': 18.},
            ticks='outside',
        ),
        margin=dict(
            t=50
        ),
        hovermode='closest',
        bargap=0.8,
        xaxis2=dict(
            domain=[AREA, 1],
            showticklabels=False
        ),
        yaxis2=dict(
            domain=[AREA, 1],
            showticklabels=False
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig = go.Figure(data=data, layout=layout)

    fig.write_image(plot_path)

def get_img_and_metadata_by_ind(img_ind):
    img_paths = ImgPathGenerator(subj_root)

    with open(os.path.join(img_paths.get_root(), "_CropPos.txt")) as f:
        cp_x, cp_y = map(int, f.readline().split(' '))

    with open(os.path.join(img_paths.get_root(), 'head_mov.txt'), 'r') as file:
        head_mov_data = [
            tuple(map(int, line.split(' ')))
            for line in file.readlines()
        ]

    data_name = 'FullSignal.csv'
    for filename in os.listdir(subj_root):
        if filename.startswith('DOT') and filename.endswith('.tsv'):
            data_name = filename

    data = pd.read_csv(
        os.path.join(subj_root, data_name),
        sep='\t'
    )

    skipped = 0
    for i, img_path in enumerate(img_paths):
        if i >= data.shape[0] or data.iloc[i][1:].isna().all():
            skipped += 1
            continue
        if i >= img_ind:
            sample_ind = img_ind - skipped
            img = imread(img_path)
            head_mov = head_mov_data[i]
            break

    return img, (cp_x, cp_y), head_mov

def get_base_and_shifted_images(img_ind, shifts):
    img, center, head_mov = get_img_and_metadata_by_ind(img_ind)

    base_img = get_shifted_crop(
        img, center, head_mov,
        { 'sh_hor': 0., 'sh_ver': 0. }
    )

    shifted_img = get_shifted_crop(
        img, center, head_mov,
        { 'sh_hor': shifts[0], 'sh_ver': shifts[1] }
    )

    return base_img, shifted_img

def get_saccade_images(img_start_ind, img_end_ind):
    img_start, center_start, head_mov_start = get_img_and_metadata_by_ind(img_start_ind)
    img_end, center_end, head_mov_end = get_img_and_metadata_by_ind(img_end_ind)

    start_crop = get_shifted_crop(
        img_start, center_start, head_mov_start,
        { 'sh_hor': 0., 'sh_ver': 0., }
    )

    end_crop = get_shifted_crop(
        img_end, center_end, head_mov_end,
        { 'sh_hor': 0., 'sh_ver': 0., }
    )
    
    return start_crop, end_crop

def changes_psog_plot(start_img, end_img, prefix):
    output_dir = os.path.join('tmp', 'psog_vis_upd')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    start_img_path = os.path.join(output_dir, prefix + '_img_start.jpg')
    end_img_path = os.path.join(output_dir, prefix + '_img_end.jpg')

    imsave(start_img_path, start_img)
    imsave(end_img_path, end_img)

    start_img = imread(start_img_path, as_gray=True)
    end_img = imread(end_img_path, as_gray=True)

    psog = PSOG()

    psog.plot_layout(start_img)
    psog.plot_layout(end_img)

    psog_plot(
        psog.simulate_output(start_img),
        os.path.join(output_dir, prefix + '_psog_start.png')
    )
    psog_plot(
        psog.simulate_output(end_img),
        os.path.join(output_dir, prefix + '_psog_end.png')
    )

    psog_plot(
        calc_normalized_psog_diff(psog, start_img, end_img),
        os.path.join(output_dir, prefix + '_psog_diff_norm.png')
    )

def shift_psog_plot(subj_root, shifts):
    base_img, shifted_img = get_base_and_shifted_images(622, shifts)
    changes_psog_plot(base_img, shifted_img, 'shifts')

def eye_pos_psog_plot(subj_root):
    start_img, end_img = get_saccade_images(622, 625)
    changes_psog_plot(start_img, end_img, 'saccade')

def shifts_distrib_plot(subj_root_etra, subj_root_randn, prefix):
    X_train_etra, _ = get_subj_data(subj_root_etra, with_shifts=True)
    X_train_randn, _ = get_subj_data(subj_root_randn, with_shifts=True)
    plot_shifts(
        X_train_etra[:, -2:],
        X_train_randn[:, -2:],
        os.path.join('tmp', 'psog_vis', prefix + '_shifts.png')
    )

if __name__ == '__main__':
    if len(sys.argv) == 4:
        etra_root = sys.argv[1]
        randn_root = sys.argv[2]
        subj = sys.argv[3]
        etra_subj_root = os.path.join(etra_root, subj)
        randn_subj_root = os.path.join(randn_root, subj)
        shifts_distrib_plot(
            etra_subj_root,
            randn_subj_root,
            'overlaid'
        )
    else:
        root = sys.argv[1]
        subj = sys.argv[2]
        subj_root = os.path.join(root, subj)

        shift_psog_plot(subj_root, (4., 0.))
        eye_pos_psog_plot(subj_root)