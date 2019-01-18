import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

def deg_to_pix(deg):
    posx, posy = deg
    dist_mm = 500.
    w_mm = 374.
    h_mm = 300.
    w_pix = 1280
    h_pix = 1024
    conv = lambda data, pix, mm, dist: \
        int(round(np.tan(data / 180. * np.pi) * dist * pix/mm + pix/2.))
    return conv(posx, w_pix, w_mm, dist_mm), \
        conv(-posy, h_pix, h_mm, dist_mm)

EET_DATA_ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\dataset\\Google project recordings\\Heatmaps_01_S_S{:03d}_R04_SHVSS3{:d}_BW_ML_120Hz\\'
BLENDER_DATA_ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\psog_nn\\eet_data_converter\\blender_data\\{}\\{}_grid15.csv'
MODULE_PREFIX = r'.\models_eval\keras_cnn'

def get_train_data(train_exp):
    import json
    from utils.sensor3 import Sensor
    ROOT = r'D:\DmytroKatrychuk\dev\research\psog_nn\eet_data_converter\blender_data'
    DESIGN = 'grid15'
    def filter_outliers(data, verbose=True):
        if verbose:
            print(data[(data.posx.abs() > 20.) 
            | (data.posy.abs() > 20)])
        return data.drop(data[(data.posx.abs() > 20.) 
            | (data.posy.abs() > 20)].index)

    with open(os.path.join('designs', 'grid15' + '.json'), 'r') as f:
        sensor = Sensor(json.load(f))

    X_train = []
    y_train = []
    for exp in train_exp:
        path = os.path.join(ROOT, exp, exp + '_%s.csv'%DESIGN)
        data = pd.read_csv(path, sep='\t', index_col=0)
   
        data = filter_outliers(data)

        X = data[sensor.sr_names].values
        y = data[['posx', 'posy']].values

        X_train.extend(X)
        y_train.extend(y)

    return np.array(X_train), np.array(y_train)

def normalize(X_train, X_test, subjs, load, mode):
    from sklearn.externals import joblib
    from sklearn.decomposition import PCA
    norm_path = os.path.join(MODULE_PREFIX, 'normalizer_' + str(subjs) + '.pkl')
    if not load:
        # save spatial information
        components = None if mode == 'cnn' else 0.99
        normalizer = PCA(n_components=components, whiten=True, random_state=0o62217)
        normalizer.fit(X_train)
        joblib.dump(normalizer, norm_path)
    else:
        normalizer = joblib.load(norm_path)
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test

def draw_distrib(subj):
    from sklearn.model_selection import train_test_split
    EET_DATA_ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\dataset\\Google project recordings\\Heatmaps_01_S_S{:03d}_R04_SHVSS3{:d}_BW_ML_120Hz\\'
    BLENDER_DATA_ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\psog_nn\\eet_data_converter\\blender_data\\{}\\{}_grid15.csv'

    subj_root = EET_DATA_ROOT.format(subj, 2 if subj < 11 else 4)

    subj = str(subj)

    data_path = 'DOT-R19.xml'
    for filename in os.listdir(subj_root):
        if filename.endswith('.xml'):
            data_path = filename

    tree = ET.parse(os.path.join(subj_root, data_path))
    root = tree.getroot()

    stimuli_pos = []
    for position in root.iter('Position'):
        x = int(position.find('X').text)
        y = int(position.find('Y').text)
        stimuli_pos.append((x, y))

    calib_pos = list(set(stimuli_pos))
    calib_pos.sort()
    calib_pos = [
        calib_pos[0],
        calib_pos[2],
        calib_pos[4],
        calib_pos[8],
        calib_pos[10],
        calib_pos[12],
        calib_pos[16],
        calib_pos[18],
        calib_pos[20]
    ]
    # stimuli_pos = list(set(stimuli_pos))
    # stimuli_pos.sort()

    from operator import itemgetter as iget
    stimuli_grid = sorted(list(set(stimuli_pos)))
    stimuli_pos = np.array(stimuli_pos)

    X_train, y_train = get_train_data([subj])

    train_ind = []
    test_ind = []
    for ind, pos in enumerate(y_train):
        posx, posy = deg_to_pix(pos)
        calib_point = False
        for calib in calib_pos:
            x, y = calib
            dist = np.hypot(posx - x, posy - y)
            if dist < 35.:
                calib_point = True
                break
        if calib_point:
            train_ind.extend([ind])
        else:
            test_ind.extend([ind])
    
    X_test = X_train[test_ind]
    y_test = y_train[test_ind]
    
    X_train = X_train[train_ind]
    y_train = y_train[train_ind]

    X_train, X_test = normalize(X_train, X_test, subj, False, 'cnn')

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

    height, width = 1024, 1280
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = [255, 255, 255]

    print('Train: ', y_train.shape)
    for pos in y_train:
        x, y = deg_to_pix(pos)
        cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

    # for pos in y_val:
    #     x, y = deg_to_pix(pos)
    #     cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
    
    print('Test: ', y_test.shape)
    for pos in y_test:
        x, y = deg_to_pix(pos)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    # workaround for correct order
    grid_order = np.array([
        1, 6, 9, 14, 17,
        2, 10, 18,
        3, 7, 11, 15, 19,
        4, 12, 20,
        5, 8, 13, 16, 21]) - 1
    for ind, grid_ind in enumerate(grid_order):
        pos = stimuli_grid[grid_ind]
        x, y = pos
        print('Stimuli grid: ', x, y)
        cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
        cv2.putText(img, str(ind + 1), (x - 7, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        if ind + 1 in [1, 3, 5, 9, 11, 13, 17, 19, 21]:
            cv2.circle(img, (x, y), 35, (0, 0, 0), 1)

    cv2.imwrite('data_distrib.jpg', img)

if __name__ == "__main__":
    draw_distrib(8)
    exit()

    for subj in range(1, 23 + 1):
        if subj == 9:
            continue
        
        subj_root = EET_DATA_ROOT.format(subj, 2 if subj < 11 else 4)

        data_path = 'DOT-R19.xml'
        for filename in os.listdir(subj_root):
            if filename.endswith('.xml'):
                data_path = filename

        tree = ET.parse(os.path.join(subj_root, data_path))
        root = tree.getroot()

        stimuli_pos = []
        for position in root.iter('Position'):
            x = int(position.find('X').text)
            y = int(position.find('Y').text)
            stimuli_pos.append((x, y))

        calib_pos = list(set(stimuli_pos))
        calib_pos.sort()
        calib_pos = [
            calib_pos[0],
            calib_pos[2],
            calib_pos[4],
            calib_pos[8],
            calib_pos[10],
            calib_pos[12],
            calib_pos[16],
            calib_pos[18],
            calib_pos[20]
        ]
        # stimuli_pos = list(set(stimuli_pos))
        # stimuli_pos.sort()

        stimuli_pos = np.array(stimuli_pos)

        height, width = 1024, 1280
        img = np.zeros((height, width, 3), np.uint8)
        img[:, :] = [255, 255, 255]

        data = pd.read_csv(BLENDER_DATA_ROOT.format(subj, subj), sep='\t')

        count = 0
        for ind, pos in enumerate(stimuli_pos):
            x, y = pos
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            # cv2.putText(img, str(ind), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            
            if (x, y) not in calib_pos:
                continue

            # for posx, posy in zip(data['posx'].values[ind*125 + 25: (ind + 1)*125 - 50],
            #         data['posy'].values[ind*125 + 25: (ind + 1)*125 - 50]):
            #     x, y = deg_to_pix((posx, posy))
            #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            #     count += 1

            for posx, posy in zip(data['posx'].values[ind*125: (ind + 1)*125],
                    data['posy'].values[ind*125: (ind + 1)*125]):
                posx, posy = deg_to_pix((posx, posy))
                dist = np.hypot(posx - x, posy - y)
                if dist > 100.:
                    pass
                cv2.circle(img, (posx, posy), 1, (0, 0, 255), -1)
                count += 1

        cv2.imwrite('stimuli.jpg', img)
        if count < 2000:
            print('Count: ', count)
        break