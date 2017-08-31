#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:34:24 2017

@author: ajaver
"""

from keras.models import load_model
import glob
import os
import tables
import numpy as np
import matplotlib.pylab as plt

SAVE_DIR = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/'

#log_dir = '/Users/ajaver/Documents/GitHub/work-in-progress/work_in_progress/skeletons_CNN/main_logs_20170321_210122'
#log_dir = '/Users/ajaver/Documents/GitHub/work-in-progress/work_in_progress/skeletons_CNN/logs/tiny_20170322_153316'

#%%
#log_dir_n = 'simple_20170321_210122'
#log_dir_n = 'resnet_20170322_191529'
#log_dir_n = 'simple_20170323_154817'
log_dir_n = 'main_20170323_153747'
#log_dir_n = 'main_20170323_204136'
#log_dir_n = 'simple_v2_20170328_171828'
is_tiny = False
#%%
#log_dir_n = 'tiny_simple_20170321_232912' #i don't think i am using the sample samples i used to train the network
#log_dir_n = 'tiny_resnet_20170322_191940'

#log_dir_n = 'tiny_20170323_170847'
#log_dir_n = 'tiny_resnet_20170322_165535_100epochs'
#log_dir_n = 'tiny_pyramid_small_20170323_172456'
#log_dir_n = 'tiny_20170324_165206'
#log_dir_n = 'tiny_20170324_174653'
#log_dir_n = 'tiny_pyramid_feat2_20170330_162429'
#is_tiny = True
#%%
sample_file = 'N2 on food R_2011_03_09__11_58_06___6___3_sample.hdf5'

log_dir = os.path.join(SAVE_DIR, 'logs', log_dir_n)
if is_tiny:
    sample_file = os.path.join(log_dir, 'input_set.hdf5')
else:
    sample_file = os.path.join(SAVE_DIR, 'data', sample_file)


rand_seed = 1337
np.random.seed(rand_seed)

with tables.File(sample_file, 'r') as fid:
    
    if is_tiny:
        #select a tiny sample
        X_set = fid.get_node('/X')
        Y_set = fid.get_node('/Y')
    
    else:
        X_set = fid.get_node('/mask')
        Y_set = fid.get_node('/skeleton')
    
    
    tot = X_set.shape[0]
    roi_size = X_set.shape[1]
    
    inds = np.random.permutation(tot)[:128]
    X = X_set[inds, :, :]
    Y = Y_set[inds, :, :]
    
    if is_tiny:
        Y = Y*roi_size/2 + roi_size/2.
        
    if X.ndim == 3:
        X = X[:, :, :, np.newaxis]
    
fnames = sorted(glob.glob(os.path.join(log_dir, '*.h5')))

fnames = fnames[-1:]
#fnames = fnames[:10]

for fname in fnames:
    print(fname)
    model = load_model(fname)
    
    Y_pred = model.predict(X)
    roi_size = X.shape[1]
    #%%
    Y_pred_c = Y_pred*roi_size/2 + roi_size/2.
    Y_c = Y
    
    mad = np.mean(np.abs(Y_pred_c-Y), axis=(1,2))
    inds = mad.argsort()[::-1]
    
    #inds = range(16)
    
    
    plt.figure()
    for ii, ind in enumerate(inds[:16]):
        plt.subplot(4,4,ii+1)
        plt.imshow(np.squeeze(X[ind]), interpolation='None', cmap='gray')
        plt.grid('off')
        
        
        plt.plot(Y_c[ind, :, 0], Y_c[ind, :, 1], '-r')
        plt.plot(Y_c[ind, 0, 0], Y_c[ind, 0, 1], 'sr')
        plt.plot(Y_pred_c[ind, :, 0], Y_pred_c[ind, :, 1], '-b')
        plt.plot(Y_pred_c[ind, 0, 0], Y_pred_c[ind, 0, 1], 'ob')
    plt.suptitle(os.path.basename(fname))

        
    
    
