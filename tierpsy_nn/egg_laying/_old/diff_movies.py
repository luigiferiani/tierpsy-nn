#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:46:17 2017

@author: ajaver
"""
import cv2
import pandas as pd
import numpy as np
import tables
import h5py
from tierpsy.analysis.ske_create.helperIterROI import generateMoviesROI, getROIfromInd
from tierpsy.analysis.compress.compressVideo import IMG_FILTERS

masked_file = '/Volumes/behavgenom_archive$/single_worm/unfinished/mutants/acr-7(tm863)II@FX863/food_OP50/XX/30m_wait/anticlockwise/acr-7 (tm863)II on food R_2010_02_24__11_23_10___2___7.hdf5'
skel_file = masked_file.replace('.hdf5', '_skeletons.hdf5')


with pd.HDFStore(skel_file, 'r') as fid:
    trajectories_data = fid['/trajectories_data']

#mask_gen = generateMoviesROI(mask_file, trajectories_data)
#for worms_in_frame in mask_gen:
#    for row_ind, img_dat in worms_in_frame.items():
#        roi_img, roi_corner = img_dat
#        
#        
#        if roi_prev is not None:
#            roi_diff = roi_img

with tables.File(masked_file, 'r') as fid:
    mask_shape = fid.get_node('/mask').shape

diff_file = 'test_diff.hdf5'

prev_img = None
dataset_diff = None
worm_index = 1
t_s = 12500
t_f = 13500

im_w = 100
im_h = 100

with h5py.File(diff_file, 'w') as fid:
    tot_frames, im_height, im_width = mask_shape
    for ii, frame_number in enumerate(range(t_s, t_f)):
        row, worm_roi, roi_corner = \
        getROIfromInd(masked_file, trajectories_data, frame_number, worm_index)
        
        
        worm_roi = worm_roi.astype(np.float)
        if prev_img is not None:
            mask = (worm_roi*prev_img) != 0
            worm_diff = np.zeros_like(worm_roi)
            worm_diff[mask] = worm_roi[mask] - prev_img[mask]
            dataset_diff[ii] = im_new = cv2.resize(worm_diff, (im_w,im_h), interpolation=cv2.INTER_CUBIC)
        else:
            #im_height, im_width = worm_roi.shape
            dataset_diff = fid.create_dataset(diff_file, 
                                     (t_f-t_s, im_w,im_h),
                                     dtype=np.float,
                                     maxshape=(None, im_w,im_h),
                                     chunks=(1, im_w,im_h),
                                     **IMG_FILTERS)
            
        
        
        prev_img = worm_roi
        
        
        if frame_number % 500 == 0:
            print(frame_number)
    
    
    
    
#%%
import matplotlib.pylab as plt
plt.imshow(worm_diff)



#    
#    masks = fid.get_node('/mask')
#    for frame in range(t_s, t_f):
#        