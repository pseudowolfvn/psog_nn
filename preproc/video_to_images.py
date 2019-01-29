import os

import av
from matplotlib import pyplot as plt
import numpy as np

ROOT = 'D:\\DmytroKatrychuk\\dev\\research\\dataset\\Google project recordings\\Heatmaps_01_S_S{:03d}_R04_SHVSS34_BW_ML_120Hz\\'

for subj in range(11, 23 + 1):
    subj_root = ROOT.format(subj)

    print("Working dir: " + subj_root)

    video_path = 'DOT-R22.avi'
    for filename in os.listdir(subj_root):
        if filename.endswith('.avi'):
            video_path = filename

    print("Found video at " + video_path)

    v = av.open(os.path.join(subj_root, video_path))

    imgs_path = os.path.join(subj_root, 'images')
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)

    count = 0
    for packet in v.demux():
        for frame in packet.decode():
            img = frame.to_image() 
            img.save(os.path.join(imgs_path, str(count) + '.jpg'))
            count += 1

    print('Video contatins ' + str(count) + ' frames')

