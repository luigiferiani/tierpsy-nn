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
from functools import partial

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from skeletons_flow import SkeletonsFlow

wild_isolates = ['JU393', 'JU402', 'ED3054', 'JU394', 
                 'N2', 'JU440', 'ED3021', 'ED3017', 
                 'JU438', 'JU298', 'JU345', 'RC301', 
                 'VC40429', 'AQ2947', 'ED3049',
                 'PS312', 'LSJ1', 'JU258', 'MY16', 
                 'CB4852', 'CB4856', 'CB4853'
                 ]


if sys.platform == 'linux':
    log_dir_root = '/work/ajaver/classify_strains/results'
    main_file = os.path.join(os.environ['TMPDIR'], 'SWDB_skel_smoothed.hdf5')
    #main_file = '/work/ajaver/classify_strains/train_set/SWDB_skel_smoothed.hdf5'
else:        
    log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains'
    main_file = '/Users/ajaver/Desktop/SWDB_skel_smoothed.hdf5'

def main(
    epochs = 5000,
    model_type = 'simple',
    is_reduced = False,
    is_wild_isolates = False,
    saving_period = None,
    model_path = None,
    n_batch = None,
    is_angle = False
    ):

    # for reproducibility
    rand_seed = 1337
    np.random.seed(rand_seed)  
    
    if is_reduced:
      bn_prefix = 'R_'
      valid_strains = ['AQ1033', 'AQ1037', 'AQ1038', 'CB1069', 'CB5', 'ED3054', 'JU438',
         'MT2248', 'MT8504', 'N2', 'NL1137', 'RB2005', 'RB557', 'VC12']
    elif is_wild_isolates:
      bn_prefix = 'W_'
      valid_strains = wild_isolates 
    else:
      bn_prefix = ''
      valid_strains = None
      
    if is_angle:
        bn_prefix += 'ang_'

    if saving_period is None:
      #the saving period must be larger in the reduce form since it has 
      #less samples per epoch
      if is_reduced:
        saving_period = 12
      else:
        saving_period = 3

    model = None
    if model_path is not None:
      assert os.path.exists(model_path)
      from keras.models import load_model
      model = load_model(model_path)
      model_type = model.name

    elif model_type == 'simple':
        from models import simple_model
        model_fun = simple_model

    elif model_type == 'larger':
        from models import larger_model
        model_fun = larger_model
        
    elif 'resnet50' in model_type:
        from models import resnet50_model
        if not '_D' in model_type:
          model_fun = resnet50_model
        else:
            dropout_rate = float(model_type.partition('_D')[-1])
            model_fun = partial(resnet50_model, dropout_rate=dropout_rate)
    else:
        ValueError('Not valid model_type')
    
    if n_batch is None:
      #resnet use more memory i  have to reduce the batch size to fit it in the GPU    
      if 'resnet50' in model_type:
        n_batch = 32
      else:
        n_batch = 64

    train_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = n_batch, 
                                   set_type='train',
                                   valid_strains = valid_strains,
                                   is_angle = is_angle
                                   )
    val_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = n_batch, 
                                   set_type='val',
                                   valid_strains = valid_strains,
                                   is_angle = is_angle
                                   )
    
    if model is None:
      #create the model using the generator size if necessary
      X,Y = next(train_generator)
      input_shape = X.shape[1:]
      output_shape = Y.shape[1:]
      model = model_fun(input_shape, output_shape)
    else:
      #TODO assert the generator and the model match sizes
      pass
    

    print(train_generator.skeletons_indexes['strain'].unique())
    print(model.summary())    
    
    base_name = bn_prefix + model.name


    log_dir = os.path.join(log_dir_root, 'logs', '%s_%s' % (base_name, time.strftime('%Y%m%d_%H%M%S')))
    pad=int(np.ceil(np.log10(epochs+1)))
    checkpoint_file = os.path.join(log_dir, '%s-{epoch:0%id}-{val_loss:.4f}.h5' % (base_name, pad))
    
    
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