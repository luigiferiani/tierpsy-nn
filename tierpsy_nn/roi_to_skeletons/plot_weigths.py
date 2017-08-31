#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:40:00 2017

@author: ajaver
"""

import os
import tables
import matplotlib.pylab as plt
import numpy as np

from keras.models import Model
from keras.models import load_model
from keras import backend as K


SAVE_DIR = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/'

model_name = os.path.join(SAVE_DIR, 'logs', 'simple_20170323_154817', 'simple-018-0.0313.h5')
model = load_model(model_name)

sample_file = 'N2 on food R_2011_03_09__11_58_06___6___3_sample.hdf5'
sample_file = os.path.join(SAVE_DIR, 'data', sample_file)

rand_seed = 1337
np.random.seed(rand_seed)

with tables.File(sample_file, 'r') as fid:
    
    X_set = fid.get_node('/mask')
    Y_set = fid.get_node('/skeleton')
    
    
    tot = X_set.shape[0]
    roi_size = X_set.shape[1]
    
    inds = np.random.permutation(tot)[:128]
    X = X_set[inds, :, :]
    Y = Y_set[inds, :, :]
    
    
    if X.ndim == 3:
        X = X[:, :, :, np.newaxis]
        
ii = 0
plt.figure()
plt.imshow(np.squeeze(X[ii]), interpolation='none', cmap='gray')
plt.grid('off')
#%%
get_layer = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.layers[10].output])
first_layer = get_layer([X,0])[0]
#%%
X_pred = first_layer[ii]
tot_maps = X_pred.shape[-1] 

col_size = min(7, np.ceil(np.sqrt(tot_maps)))
tot_subplots = col_size*col_size
for ires in range(tot_maps):
    if ires % tot_subplots == 0:
        plt.figure()
    
    plt.subplot(col_size,col_size, (ires%tot_subplots)+1)
    img = np.squeeze(X_pred[:,:,ires])
    plt.imshow(img, interpolation='none', cmap='gray')
    plt.grid('off')
    plt.axis('off')
    
    

    



#%%