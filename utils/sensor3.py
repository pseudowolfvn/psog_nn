#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: rz
@email: zemblysr@msu.edu
"""
#%% imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['image.cmap'] = 'gray'

from PIL import Image
from tqdm import tqdm

#%% functions and constants

def run_feature_extraction(exp_path, sensor, fpath_sr=None):
    '''Runs feature extraction aka calculates sensor responses

    '''
    exp_dir, _ = os.path.split(exp_path)
    spath_tmpl = '{:06d}_{:+.2f}_{:+.2f}_{:+.2f}_{:+08.4f}_{:+08.4f}.png'

    #read meta data
    data_exp = pd.read_csv(exp_path, sep='\t')
    data_exp.rename(columns={'movCam_H': 'smh',
                             'movCam_D': 'smr',
                             'movCam_V': 'smv'}, inplace=True)
    sr = []
    #TODO: multitheading
    for n, frame in tqdm(data_exp.iterrows()):
        _movCam = frame[['smh', 'smr', 'smv']].values.tolist()
        _pt = frame[['posx', 'posy']].values.tolist()
        #get file name
        fname = spath_tmpl.format(n ,*(_movCam+_pt))
        img = Image.open('%s/%s' % (exp_dir, fname))
        img = np.array(img.convert('L'))/255.
        #get sensor responses

        _sr, _ = sensor.getSR(img)
        sr.append(_sr)
#        if n>100:
#            break
    sr_ = pd.DataFrame(sr, columns=sensor.sr_names)
    data_exp = pd.concat([data_exp, sr_], axis=1)

    #save feature file
    if not(fpath_sr is None):
        data_exp.to_csv(fpath_sr, sep='\t')

    return data_exp

def plot_design(exp_path, n, sensor, plot_type=2):
    '''
    Convenience function to plot design

    '''
    exp_dir, _ = os.path.split(exp_path)
    spath_tmpl = '{:06d}_{:+.2f}_{:+.2f}_{:+.2f}_{:+08.4f}_{:+08.4f}.png'

    data_exp = pd.read_csv(exp_path, sep='\t')
    data_exp.rename(columns={'movCam_H': 'smh',
                             'movCam_D': 'smr',
                             'movCam_V': 'smv'}, inplace=True)
    frame = data_exp.iloc[n]
    _movCam = frame[['smh', 'smr', 'smv']].values.tolist()
    _pt = frame[['posx', 'posy']].values.tolist()
    fname = spath_tmpl.format(n ,*(_movCam+_pt))
    img = Image.open('%s/%s' % (exp_dir, fname))
    img = np.array(img.convert('L'))/255.
    sensor.getSR(img, plot=plot_type)

def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and sigma of sig
    """
    #l=r*2+1
    #sig=(r*2+1)/4.
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    k = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    kernel = (k-np.max(k))/np.ptp(k)+1
    return kernel# / np.sum(kernel)# * 255

class Sensor():
    '''Implements sensor class
    '''
    def __init__(self, design):
        self.coord = design['coord_space']
        self.scs = None #sensor corner coordinates
        self.swh = None #sensor width and height
        self.scc = None #sensor center coordinates
        self.ob = 0     #padding

        #calculate kernels
        #TODO: automatically calculate kernels for each sensor
        kernels = [gkern(r*2+1, (r*2+1)/4.) for _, (r, _), _ in design['design']]

        #design = (type, (radius_x, radius_y), (r, theta))
        self.design = list(zip(*design['design']))
        self.design.append(kernels)
        self.design = list(zip(*self.design))

        #if defined, load output transitions. Else use raw outputs
        self.output_transition = np.array(design['output']) if len(design['output'])\
                                                            else np.eye(len(self.design))
        self.sr_names = list(map(lambda x: 'sr_%d'%x, range(len(self.output_transition))))

    def getSC(self, img, image_center_offset=[0, 0]):
        '''
        get sensor coordinates
        '''
        self.img = img
        img_shape = np.array(self.img.shape[:2][::-1])
        image_center = img_shape/2+image_center_offset

        #TODO: check for incorrect coordinate system
        if self.coord == 'polar':
            #convert polar to cartesian
            sensor_centers = [
                [round(r*np.cos(np.deg2rad(theta))),
                 round(r*np.sin(np.deg2rad(theta)))]
                for _, _, (r, theta), _ in self.design
            ]
        else:
            sensor_centers = [
                [coord_x, coord_y]
                for _, _, (coord_x, coord_y), _ in self.design
            ]

        #absolute coordinates of sensor corners
        sensor_coordinates_c = np.int32(image_center + sensor_centers)
        _, sensor_sizes, _, kernels = zip(*self.design)
        sensor_coordinates_s = sensor_coordinates_c - sensor_sizes

        #check off boundary
        swh = np.array(sensor_sizes)*2+1
        sensor_coordinates_e = sensor_coordinates_s + swh
        _sce = img_shape - sensor_coordinates_e

        #pad image if sensors "see" outside of image
        if (sensor_coordinates_s < 0).any() or (_sce < 0).any():
            ob = min([sensor_coordinates_s.min(), _sce.min()])
            sensor_coordinates_s-=ob
            sensor_coordinates_c-=ob
            self.ob  = abs(int(ob))
            self.img = np.pad(self.img, self.ob, mode = 'reflect')

        return sensor_coordinates_s, swh, sensor_coordinates_c, kernels

    def getSR(self, img, image_center_offset=[0, 0], reuse_sensor_placement=False, plot=False):
        '''
        get sensor response
        '''

        if reuse_sensor_placement and not any([_sc is None for _sc in [self.scc, self.swh, self.scs]]):
            #experimantal. not used
            scs, swh, scc, kernels = self.scs, self.swh, self.scc, self.kernels
            self.img = img
        else:
            scs, swh, scc, kernels = self.getSC(img, image_center_offset)
            self.scs, self.swh, self.scc, self.kernels = scs, swh, scc, kernels

        output = []
        img_patches = []
        img_patches_f = []

        for _sc, (_w, _h), kernel in zip (scs, swh, kernels):
            #crop image
            _img = self.img[_sc[1]:_sc[1]+_h, _sc[0]:_sc[0]+_w].copy()
            #calculate output
            _img_f = _img * kernel
            img_patches.append(_img)
            img_patches_f.append(_img_f),
            #This replicates Rigas et al, 2017
            output.append(np.mean(_img_f[_img_f>0]))
            #TODO:
            #output.append(np.mean(_img_f))
        output_raw = np.array(output)
        #print output_raw
        #calculate output defined in design
        output = [np.sum(output_raw*_o) for _o in self.output_transition]

        #plot
        if plot:
            #plots sensor placement
            img = self.img.copy()
            if plot == 1:
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(img)
                img_sr = img*0.1

                for n, (_sc, (_w, _h), _sct, _img_f) in enumerate(zip(scs, swh, scc, img_patches_f)):
                    ax[0].add_patch(patches.Rectangle(_sc, _w, _h, fill=False))
                    img_sr[_sc[1]:_sc[1]+_h, _sc[0]:_sc[0]+_w]+=_img_f
                    ax[1].add_patch(patches.Circle(_sct, _w/4, fill=False))
                    ax[1].text(*_sct, s=n, ha='center', va='center', color='red')
                ax[1].imshow(img_sr)
            if plot==2:
                fig = plt.figure(figsize=[3.2, 2.8])
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                _h, _w = img.shape

                ax.imshow(img)
                ax.set_xlim(self.ob, _w-self.ob)
                ax.set_ylim(self.ob, _h-self.ob)
                plt.gca().invert_yaxis()

                for n, (_sc, (_w, _h), _sct, _img_f) in enumerate(zip(scs, swh, scc, img_patches_f)):
                    # manually calculated half reception angle.
                    # applies only for window patch of 121 px
                    _fwhm = 70
                    ax.add_patch(patches.Circle(_sct, _fwhm/2., fill=False))
                    ax.text(*_sct, s=n+1, ha='center', va='center', color='red')

                    #custom plot for ligaze
#                    plt.gca().add_patch(patches.Circle(_sct, 5., fill=False))


        return output, (scs, img_patches, img_patches_f)