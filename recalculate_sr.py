import json
import os
import sys

import numpy as np
from PIL import Image
from utils.sensor import Sensor, run_feature_extraction


DESIGN = 'grid15'


def calculate_blender_sensor_outputs(root, exp=None, img_ext='.png', recalc=False):
    #%%load test data
    design_path = 'designs\{}.json'.format(DESIGN)
    print(design_path)
    with open(design_path, 'r') as f:
        sensor = Sensor(json.load(f))

    data_exp = {}

    exps = [exp if exp is not None else os.listdir(root)]

    for exp in exps:
        exp_path = '{root}\{exp}\{exp}.csv'.format(root=root, exp=exp)
        fpath_sr = exp_path.replace('.csv', '_%s.csv'%DESIGN)
        print(fpath_sr)
        print(exp_path)
        data_exp[exp] = run_feature_extraction(exp_path, sensor, fpath_sr=fpath_sr)


def calculate_eet_sensor_outputs(root, img_ext='.jpg', recalc=False):
    design_path = 'designs\{}.json'.format(DESIGN)
    print(design_path)
    with open(design_path, 'r') as f:
        sensor = Sensor(json.load(f))

    for dirname in os.listdir(root):
        if not os.path.isdir(os.path.join(root, dirname)):
            continue
        csv_name = int(dirname.split('_')[3][1:])
        exp_path = os.path.join(root, dirname, 'images_centered_shifted', str(csv_name) + '.csv')
        if not os.path.exists(exp_path):
            continue
        fpath_sr = exp_path.replace('.csv', '_%s.csv'%DESIGN)
        if os.path.exists(fpath_sr) and not recalc:
            print(fpath_sr)
            print('File already exists, skipping...')
            continue
        print(fpath_sr)
        print(exp_path)
        run_feature_extraction(exp_path, sensor, fpath_sr=fpath_sr, img_mode='eet')


BLENDER_DATA_ROOT = r'.\eyelink_data_converter\blender_data'
UNITYEYES_DATA_ROOT = r'..\UnityEyes_Windows'
EET_DATA_ROOT = r'D:\DmytroKatrychuk\dev\research\dataset\Google project recordings'


if __name__ == '__main__':
    if sys.argv[1] == 'blender':
        calculate_blender_sensor_outputs(BLENDER_DATA_ROOT, 9)
    elif sys.argv[1] == 'unityeyes':
        calculate_blender_sensor_outputs(UNITYEYES_DATA_ROOT, 'imgs_125k_resized', '.jpg')
    elif sys.argv[1] == 'eet':
        calculate_eet_sensor_outputs(EET_DATA_ROOT)
    elif sys.argv[1] == 'plot':
        design_path = 'designs\{}.json'.format(DESIGN)
        with open(design_path, 'r') as f:
            sensor = Sensor(json.load(f))
        imgs_path = os.path.join(EET_DATA_ROOT, 'images_centered_shifted')
        for filename in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, filename)
            img = Image.open(img_path)
            img = np.array(img.convert('L'))/255.
            sensor.getSR(img, plot=1)
            break