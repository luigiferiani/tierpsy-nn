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
    n_batch = 64,
    model_type = 'simple',
    is_reduced = True
    ):
    
    if is_reduced:
      valid_strains = ['AQ1033', 'AQ1037', 'AQ1038', 'CB1069', 'CB5', 'ED3054', 'JU438',
         'MT2248', 'MT8504', 'N2', 'NL1137', 'RB2005', 'RB557', 'VC12']
    else:
      valid_strains = None

    # for reproducibility
    rand_seed = 1337
    np.random.seed(rand_seed)  
    
    
    train_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = n_batch, 
                                   set_type='train',
                                   valid_strains = valid_strains
                                   )
    val_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = n_batch, 
                                   set_type='val',
                                   valid_strains = valid_strains
                                   )
    print(train_generator.skeletons_indexes['strain'].unique())

    X,Y = next(train_generator)
    input_shape = X.shape[1:]
    output_shape = Y.shape[1:]
    
    if model_type == 'simple':
        from models import simple_model
        model = simple_model(input_shape, output_shape)
    elif model_type == 'larger':
        from models import larger_model
        model = larger_model(input_shape, output_shape)
    else:
        ValueError('Not valid model_type')

    base_name = model.name
    if is_reduced:
      base_name = 'R_' + base_name 

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