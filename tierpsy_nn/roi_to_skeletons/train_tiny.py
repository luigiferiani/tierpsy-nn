#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:39:21 2017

@author: ajaver
"""
import tables
import numpy as np
import matplotlib.pylab as plt
import time
import os

from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, History
from keras.optimizers import Adam

from params import SAVE_DIR
from skelxy2ang import transform2skelangles

import tested_models

#%%
if __name__ == '__main__':
    if True:
        from tested_models import test_simple_head
        
        model_func = tested_models
        
        sample_file = 'N2 on food R_2011_03_09__11_58_06___6___3_sample.hdf5'
        
        sample_file = os.path.join(SAVE_DIR, 'data', sample_file)
        with tables.File(sample_file, 'r') as fid:
            #select a tiny sample
            tot = fid.get_node('/mask').shape[0]
            inds = np.random.permutation(tot)[:128]
            X = fid.get_node('/mask')[inds, :, :][:, :, :, np.newaxis]
            Y = fid.get_node('/skeleton')[inds, :, :]
            roi_size = X.shape[1]
            
            #Y = Y/roi_size
            
            Y = (Y-(roi_size/2.))/roi_size*2
            #X = -(X-np.mean(X, axis=(1,2)))
            
            
            Y = Y[:, 0, :]
            
            #skel_angles, mean_angles, segment_sizes, ini_coord = transform2skelangles(Y)
            #Y = np.concatenate([skel_angles, mean_angles[:,np.newaxis], segment_sizes[:,np.newaxis], ini_coord], axis=1)
            
        model, model_name = test_simple_head(Y.shape[1:])
        
        
        
    else:
        #model_dir = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/logs/tiny_pyramid_short_a_20170324_174653'
        #model = load_model(os.path.join(model_dir, 'tiny-1999-0.0388.h5'))
        
        model_name = 'tiny_pyramid_feat2_20170330_162429_v'
        model_path = '/Volumes/behavgenom_archive$/Avelino/skeletons_cnn_tests/logs/tiny_pyramid_feat2_20170330_162429/tiny-pyramid_feat2-1999-0.0760.h5'
        model = load_model(model_path)
        
        optimizer = Adam(lr=1e-6, decay=0.1)
        model.compile(loss='mean_absolute_error',
                          optimizer=optimizer,
                          metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])
        
        sample_file = os.path.join(os.path.dirname(model_path), 'input_set.hdf5')
        with tables.File(sample_file, 'r') as fid:
            #select a tiny sample
            X_set = fid.get_node('/X')
            Y_set = fid.get_node('/Y')
            tot = X_set.shape[0]
            roi_size = X_set.shape[1]
            
            X = X_set[:]
            Y = Y_set[:]
    
    #%%
    epochs = 2000
    save_period = 50
    #epochs = 100
    #save_period = 10
    
    pad=int(np.ceil(np.log10(epochs+1)))
    log_dir = os.path.join(SAVE_DIR, 'logs', '%s_%s' % (model_name, time.strftime('%Y%m%d_%H%M%S')))
    checkpoint_file = os.path.join(log_dir, 'tiny-%s-{epoch:0%id}-{loss:.4f}.h5' % (model_name, pad))
    
    history = History()
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    mcp = ModelCheckpoint(checkpoint_file, verbose=0,  mode='auto', period=save_period)
    
    os.makedirs(log_dir)
    input_data_f = os.path.join(log_dir, 'input_set.hdf5')
    with tables.File(input_data_f, 'w') as fid:
        fid.create_carray('/', 'X', obj = X)
        fid.create_carray('/', 'Y', obj = Y)
    
    
    model.fit(X,Y, batch_size=128, 
              epochs=epochs, 
              verbose=1, callbacks=[tb, mcp, history])
    
        
