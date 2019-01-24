import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from skimage.io import imread, imsave

from utils.img_path_gen import ImgPathGenerator
from utils.utils import repeat_up_to

def add_sensor_shifts(data, hor, ver):
    shift_pairs = np.array([(h, v) for h in hor for v in ver])
    shifts = repeat_up_to(shift_pairs, len(data.index))

    data['hor_shift'] = shifts[:,0] 
    data['ver_shift'] = shifts[:,1]
    return data

def shift_mm_to_pix(sh):
    STEP = 0.5
    PIX_TO_MM = 4
    return round(sh / STEP) * PIX_TO_MM

def get_shifted_crop(img, top_left, head_mov, sample):
    smh, smv = sample[['smh', 'smv']]
    x, y = top_left
    x += shift_mm_to_pix(smh) + head_mov[0]
    y += shift_mm_to_pix(smv) + head_mov[1]
    w, h = 320, 240
    if x + h // 2 > img.shape[0] or y + w // 2 > img.shape[1]:
        print('WARNING: crop out of the range!')
    # return img.crop((y - w // 2, x - h // 2, y + w // 2, x + h // 2))
    return img[x - h // 2: x + h // 2 + 1, y - w // 2: y + w // 2 + 1]

def rename(data):
    data = data.rename(index=str,
        columns={'GazePointXLeft': 'posx',
            'GazePointYLeft': 'posy',
            'hor_shift': 'smh',
            'ver_shift': 'smv',
            'PupilArea': 'pupil_size'}
    )
    data = data[['posx', 'posy'
        , 'smh', 'smv', 'pupil_size']]
    
    return data

def preprocess_subj(subj_root):
    print("Preprocessing for subject: " + subj_root)

    img_paths = ImgPathGenerator(subj_root)

    output_dir = os.path.join(subj_root, 'images_crop')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(img_paths.get_root(), "_CropPos.txt")) as f:
        cp_x, cp_y = map(int, f.readline().split(' '))

    data = pd.read_csv(
        os.path.join(subj_root, 'FullSignal.csv'),
        sep='\t'
    )

    shift_range = np.arange(-2., 2. + 0.1, 0.5)
    data = add_sensor_shifts(data, shift_range, shift_range)

    data = rename(data)

    data_name = Path(subj_root).name + '.csv'
    data.to_csv(
        os.path.join(output_dir, data_name),
        sep='\t',
        index=False
    )

    with open(os.path.join(img_paths.get_root(), 'head_mov.txt'), 'r') as file:
        head_mov_data = [tuple( map( int, line.split(' ') ) )
            for line in file.readlines()]
    
    for i, img_path in enumerate(img_paths):
        img = imread(img_path)
        img = get_shifted_crop(img,
            (cp_x, cp_y),
            head_mov_data[i],
            data.iloc[i]
        )
        img_name = Path(img_path).name
        imsave(os.path.join(output_dir, img_name), img)


def preprocess(dataset_root):
    for dirname in os.listdir(dataset_root):
        subj_root = os.path.join(dataset_root, dirname)
        preprocess_subj(subj_root)

if __name__ == "__main__":
    preprocess(sys.argv[1])