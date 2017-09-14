#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:28:03 2017

@author: ajaver
"""
import numpy as np
import os
import time
import sys

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from skeletons_flow import SkeletonsFlow


if sys.platform == 'linux':
    log_dir_root = '/work/ajaver/classify_strains/results'
    main_file = '/work/ajaver/classify_strains/train_set/SWDB_skel_smoothed.hdf5'
else:        
    log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains'
    main_file = '/Users/ajaver/Desktop/SWDB_skel_smoothed.hdf5'

def main(
    epochs = 5000,
    saving_period = 3,
    n_batch = 128,
    model_type = 'simple'
    ):
    
    # for reproducibility
    rand_seed = 1337
    np.random.seed(rand_seed)  
    
    
    train_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = n_batch, 
                                   set_type='train'
                                   )
    val_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = n_batch, 
                                   set_type='test'
                                   )
    
    X,Y = next(train_generator)
    input_shape = X.shape[1:]
    output_shape = Y.shape[1:]
    
    if model_type == 'simple':
        from models import simple_model
        model = simple_model(input_shape, output_shape)
    else:
        ValueError('Not valid model_type')

    base_name = model.name
    log_dir = os.path.join(log_dir_root, 'logs', '%s_%s' % (base_name, time.strftime('%Y%m%d_%H%M%S')))
    pad=int(np.ceil(np.log10(epochs+1)))
    checkpoint_file = os.path.join(log_dir, '%s-{epoch:0%id}-{loss:.4f}.h5' % (base_name, pad))
    
    
    tb = TensorBoard(log_dir = log_dir)
    mcp = ModelCheckpoint(checkpoint_file, 
                          monitor='loss',
                          verbose=1,  
                          mode='auto', 
                          period = saving_period
                          )

    model.compile(optimizer = Adam(lr=1e-3), 
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    
    model.fit_generator(train_generator,
                        steps_per_epoch = len(train_generator)/n_batch, 
                        epochs = epochs,
                        validation_data = val_generator,
                        validation_steps = len(val_generator)/n_batch,
                        verbose = 1,
                        callbacks=[tb, mcp]
                        )
import fire
if __name__ == '__main__':
    fire.Fire(main)