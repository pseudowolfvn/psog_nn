import os

import pandas as pd
from PIL import Image

from eyelink_data_converter.to_blender import gen_sensor_shifts


def skip_pc(x, y):
    return x < 181 or x > 459 or y < 141 or y > 207


def rename_to_blender(data):
    blender_data = data.rename(index=str, columns={
        'GazePointXLeft': 'posx',
        'GazePointYLeft': 'posy',
        'hor_shift': 'smh',
        'dep_shift': 'smd',
        'ver_shift': 'smv',
        'PupilArea': 'pupil_size'
    })
    blender_data = blender_data[['posx', 'posy'
        , 'smh', 'smd', 'smv', 'pupil_size']]

    blender_data['pupil_size'] *= 2. / 1000 # ?
    
    return blender_data


def shift_mm_to_pix(sh):
    return round(sh / 0.5) * 5


def get_img_name(ind, data):
    posx, posy, smh, smv = data[['posx', 'posy', 'smh', 'smv']]
    tmpl = '{:06d}_{:+.2f}_{:+.2f}_{:+.2f}_{:+08.4f}_{:+08.4f}.jpg'
    return tmpl.format(ind, smh, 0., smv, posx, posy)


def get_shifted_crop(img, center, data):
    smh, smv = data[['smh', 'smv']]
    x, y = center
    x += shift_mm_to_pix(smh)
    y += shift_mm_to_pix(smv)
    w, h = 320, 240
    return img.crop((x - w//2, y - h//2, x + w//2, y + h//2))



EET_DATA_ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\dataset\\Google project recordings\\Heatmaps_01_S_S{:03d}_R04_SHVSS3{:d}_BW_ML_120Hz\\'

def convert_to_blender():
    for subj in range(1, 23 + 1):
        subj_root = EET_DATA_ROOT.format(subj, 2 if subj < 11 else 4)

        print("Working dir: " + subj_root)

        input_dir = os.path.join(subj_root, 'images')
        output_dir = os.path.join(subj_root, 'images_pc_centered_shifted')

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with open(os.path.join(input_dir, "_pc.txt")) as f:
            line = f.readline()
            cr_x, cr_y = map(int, line.split(' '))
            if skip_pc(cr_x, cr_y):
                print('skip ' + subj_root)
                continue
        
        data_path = 'DOT-R22.tsv'
        for filename in os.listdir(subj_root):
            if filename.endswith('.tsv'):
                data_path = filename

        eet_data = pd.read_csv(os.path.join(subj_root, data_path), sep='\t')

        data = gen_sensor_shifts(eet_data, 
            [-2.0, -1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5, 2.0],
            [-2.0, -1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5, 2.0])

        blender_data = rename_to_blender(data)

        blender_data.to_csv(os.path.join(output_dir, str(subj) + '.csv')
            , sep='\t', index=False)

        n_samples = blender_data.shape[0]
        ind = 0
        for filename in os.listdir(input_dir):
            if 'NaN' in filename:
                continue
            if ind >= n_samples:
                break
            img = Image.open(os.path.join(input_dir, filename))
            sample = blender_data.iloc[ind]
            img_name = get_img_name(ind, sample)
            img = get_shifted_crop(img, (cr_x, cr_y), sample)
            img.save(os.path.join(output_dir, img_name))
            ind += 1
        break

if __name__ == "__main__":
    convert_to_blender()