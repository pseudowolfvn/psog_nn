import os

import pandas as pd
from PIL import Image

from eyelink_data_converter.to_blender import gen_sensor_shifts


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
    return round(sh / 0.5) * 4


def get_img_name(ind, data):
    posx, posy, smh, smv = data[['posx', 'posy', 'smh', 'smv']]
    tmpl = '{:06d}_{:+.2f}_{:+.2f}_{:+.2f}_{:+08.4f}_{:+08.4f}.jpg'
    return tmpl.format(ind, smh, 0., smv, posx, posy)


def get_shifted_crop(img, top_left, head_mov, data):
    smh, smv = data[['smh', 'smv']]
    x, y = top_left
    x += shift_mm_to_pix(smh) + head_mov[0]
    y += shift_mm_to_pix(smv) + head_mov[1]
    w, h = 320, 240
    if x + w // 2 > 640 or y + h // 2 > 348:
        print('crop out of the range!')
    return img.crop((x - w//2, y - h//2, x + w//2, y + h//2))



EET_DATA_ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\dataset\\Google project recordings\\Heatmaps_01_S_S{:03d}_R04_SHVSS3{:d}_BW_ML_120Hz\\'

def convert_to_blender():
    for subj in range(23, 23 + 1):
        subj_root = EET_DATA_ROOT.format(subj, 2 if subj < 11 else 4)

        print("Working dir: " + subj_root)

        input_dir = os.path.join(subj_root, 'images')
        output_dir = os.path.join(subj_root, 'images_nn')

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with open(os.path.join(input_dir, "_pc.txt")) as f:
           line = f.readline()
           pc_y, pc_x = map(int, line.split(' '))
        
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
        with open(os.path.join(input_dir, 'head_mov.txt'), 'r') as head_mov_file:
            head_mov_data =  [tuple(map(int, line.split(' ')))
                for line in head_mov_file.readlines()]
        img_ind = 0
        data_ind = 0
        head_data_ind = 0
        m = 0
        for x, y in head_mov_data:
            m = max(m, y)
        print(348 - 120 - 16 - 1 - m)
        while True:
            img_name = str(img_ind) + '.jpg'
            img_nan_name = str(img_ind) + '_NaN.jpg'
            if os.path.exists(os.path.join(input_dir, img_nan_name)):
                img_ind += 1
                continue
            
            fullname = os.path.join(input_dir, img_name)

            if not os.path.exists(fullname):
                print(fullname, ' doesn\'t exist')
                break

            if data_ind >= n_samples:
                print(data_ind, ' raw doesn\'t exist in csv')
                break

            if head_data_ind >= len(head_mov_data):
                 print(head_data_ind, ' raw doesn\'t exist in head_mov file')
                 break


            img = Image.open(fullname)
            sample = blender_data.iloc[data_ind]
            img_name = get_img_name(img_ind, sample)
            head_mov = head_mov_data[head_data_ind]
            img = get_shifted_crop(img, (pc_x, pc_y), head_mov, sample)
            img.save(os.path.join(output_dir, img_name))
            
            img_ind += 1
            data_ind += 1
            head_data_ind += 1
        
        head_mov_file.close()
        break

if __name__ == "__main__":
    convert_to_blender()