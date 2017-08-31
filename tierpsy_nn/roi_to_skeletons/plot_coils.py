#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:40:00 2017

@author: ajaver
"""

import tables
import numpy as np
from keras.models import load_model
import matplotlib.pylab as plt
import os


#model_name = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/logs/simple_20170321_210122/main-18-0.03.h5'
#model_name = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/logs/resnet_20170322_191529/tiny-018-0.0415.h5'
#model_name = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/logs/simple_20170323_154817/simple-010-0.0329.h5'
#model_name = '/Volumes/behavgenom_archive\$/Avelino/skeletons_cnn_tests/logs/main_20170323_153747/tiny-003-0.0357.h5'
model_name = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/logs/simple_v2_20170328_171828/simple_v2-033-0.0713.h5'

#sample_file = 'N2 on food R_2011_03_09__11_58_06___6___3_sample_bad.hdf5'
#sample_file = 'egl-43 (n1079)II on food L_2010_07_09__11_04___3___2_sample.hdf5'
sample_file = 'HW_worm3_F1-3_Set4_Pos4_Ch3_26012017_153655_sample.hdf5'

SAVE_DIR = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/'
sample_file_f = os.path.join(SAVE_DIR, 'data', sample_file)

with tables.File(sample_file_f, 'r') as fid:
    #select a tiny sample
    tot = fid.get_node('/mask').shape[0]
    inds = np.random.permutation(tot)[:128]
    #inds = np.arange(128)
    X = fid.get_node('/mask')[inds, :, :][:, :, :, np.newaxis]
    Y = fid.get_node('/skeleton')[inds, :, :]
    roi_size = X.shape[1]
    
model = load_model(model_name)

Y_pred = model.predict(X)
roi_size = X.shape[1]
#%%
#Y_c = Y*roi_size/2 + roi_size/2.
Y_pred_c = Y_pred*roi_size/2 + roi_size/2.
Y_c = Y
#Y_pred_c = Y_pred


ind = 20

for ind in range(Y_pred_c.shape[0]):
    if ind % 16 == 0:
        plt.figure()
    
    
    plt.subplot(4,4,(ind%16)+1)
    plt.imshow(np.squeeze(X[ind]), interpolation='None', cmap='gray')
    plt.grid('off')
    
    
    plt.plot(Y_c[ind, :, 0], Y_c[ind, :, 1], '-r')
    plt.plot(Y_c[ind, 0, 0], Y_c[ind, 0, 1], 'sr')
    plt.plot(Y_pred_c[ind, :, 0], Y_pred_c[ind, :, 1], '-b')
    plt.plot(Y_pred_c[ind, 0, 0], Y_pred_c[ind, 0, 1], 'ob')
    
    
    if ind > 128:
        break