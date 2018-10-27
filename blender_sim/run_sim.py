##!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:27:25 2017

@author: raimondas
"""
#%% imports
import os, sys, glob, time
import itertools
from distutils.dir_util import mkpath
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

import seaborn as sns
sns.set_style("ticks")

###
import argparse, json

import eyemodel
'''
   ^
   |    .-.
   |   |   | <- Head
   |   `^u^'
 Y |      ¦V <- Camera    (As seen from above)
   |      ¦
   |      ¦
   |      o <- Target

     ----------> X

 +X = left
 +Y = back
 +Z = up
'''

#%% functions
def get_calibration_targets(config, plot=False):
    config_calib = config['calibration']
    pupil_radius = config['eye']['pupil_radius']
    n, extent = config_calib['n_points'], config_calib['extent']

    #generate calibration points
    x_pos = np.linspace(-extent[0], extent[0], n[0])
    x_shift = extent[0]/(n[0]-1.)
    y_pos = np.linspace(-extent[1], extent[1], n[1])
    xy_pos = np.meshgrid(x_pos, y_pos)
    xy_pos[0][1::2]+=x_shift
    pt_cal = np.vstack(map(np.ravel, xy_pos)).T
    pt_cal = pt_cal[abs(pt_cal[:,0])<=extent[0]].tolist()

    #generate sensor movement vectors
    #camera movements are in mm
    config_movCam = config_calib['sensor_movement']
    step = config_movCam['step']
    movCam_H = arange_(*config_movCam['movCam_H'], step=step)
    movCam_V = arange_(*config_movCam['movCam_V'], step=step)
    movCam_D = arange_(*config_movCam['movCam_D'], step=step)
    movCam = list(itertools.product(movCam_H, movCam_D, movCam_V))

    #et = [tuple(_pt)+_cm for _pt, _cm in itertools.product(pt_cal, movCam)]
    et = list(itertools.product(pt_cal, movCam, [pupil_radius]))

    if plot:
        pass

    return et

ROOT = r'D:\DmytroKatrychuk\dev\research\psog_dk'
SOURCE = 'test_data'
DESIGN = 'jazz'
def get_arguments():
    parser = argparse.ArgumentParser(description='PSOG simulations')
    parser.add_argument('--root', type=str, default=ROOT,
                        help='Root directory of eye-movement data.')
    parser.add_argument('--source', type=str, default=SOURCE,
                        help='Source directory of eye-movement data.')
    parser.add_argument('--design', type=str, default=DESIGN,
                        help='Sensor placement design')
    return parser.parse_args()

#%% Setup parameters
args = get_arguments()
with open('config.json', 'r') as f:
    config = json.load(f)
with open('designs/%s.json'%args.design, 'r') as f:
    design = json.load(f)

def arange_(start, stop, step, endpoint=True):
    stop = stop+step if endpoint else stop
    return np.arange(start, stop, step)

if args.source=='calibration':
    et = get_calibration_targets(config)
else:
    et = pd.read_csv('{root}\\{exp}\\{exp}.csv'.format(root=args.root,
                                                     exp=args.source), sep='\t')
    for x in et.iterrows():
        _, y = x
        print(y)
    et = [[(_px, py), (_cx, cd, cy), _p/2]
          for _, (_px, py, _cx, cd, cy, _p) in et.iterrows()]

#t, _, _ = zip(*et)
#t=np.array(t)
#plt.plot(t[:,0], t[:,1], '.')
#stop
#%%prepare rendering
config_eye = config['eye']
config_camera = design['camera']
config_led = design['led']

INIT_CamPos, INIT_CamTrg = config_camera['params']
INIT_SunLoc, INIT_SunTrg = design['sun']['params']
INIT_LEDLoc, INIT_LEDTrg = zip(*config_led['leds'])

led_strength = config_led['strength']

#%% render images
with eyemodel.Renderer() as r:
    # Initialize eye parameters
    r.eye_closedness = config_eye['eye_closedness']
    r.eye_radius = config_eye['eye_radius']
    r.pupil_radius = config_eye['pupil_radius']
    r.iris = config_eye['iris']


    # Initialize camera properties
    r.image_size = config_camera['image_size']
    focal_length = (r.image_size[0]/2.0) / np.tan(45*np.pi/180 / 2)
    r.focal_length = focal_length
    r.render_samples = config_camera['render_samples']

    # Initialize ambient light
    sun = eyemodel.Light(
                type="sun",
                location = INIT_SunLoc,
                target = INIT_SunTrg,
                strength = design['sun']['strength']
                )

    for _n, (_pt, _movCam, _pr) in enumerate(tqdm(et)):
        #stop
        #mkpath(sdir)

        #move sensor
        r.camera_position = [_p_init + _p_shif for _p_init, _p_shif in zip(INIT_CamPos, _movCam)]
        r.camera_target = [_p_init + _p_shif for _p_init, _p_shif in zip(INIT_CamTrg, _movCam)]

        led_lights = [
            eyemodel.Light(
                strength = led_strength,
                location = [_p_init + _p_shif for _p_init, _p_shif in zip(INIT_Loc, _movCam)],
                target = [_p_init + _p_shif for _p_init, _p_shif in zip(INIT_Trg, _movCam)])
            for INIT_Loc, INIT_Trg in zip(INIT_LEDLoc, INIT_LEDTrg)
        ]

        r.lights = led_lights + [sun]

        #move gaze
        r.eye_target = [np.tan(np.deg2rad(_pt[0]))*-1000, -1000, np.tan(np.deg2rad(_pt[1]))*1000]
        r.pupil_radius = _pr
        #render image

        spath = '%s/%s' % (args.root, args.source)
        spath = '{}/{:06d}_{:+.2f}_{:+.2f}_{:+.2f}'.format(spath, _n, *tuple(_movCam))
        spath = "{spath}_{:+08.4f}_{:+08.4f}.png".format(*tuple(_pt), spath=spath)

        if os.path.exists(spath):
            print("skipping")
            continue

        r.render(spath, cuda=True)
        #stop

