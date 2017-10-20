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
import math
from functools import partial

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from skeletons_flow import SkeletonsFlow, wild_isolates_WT2, CeNDR_base_strains

wild_isolates_old = ['JU393', 'JU402', 'ED3054', 'JU394', 
                 'N2', 'JU440', 'ED3021', 'ED3017', 
                 'JU438', 'JU298', 'JU345', 'RC301', 
                 'VC40429', 'AQ2947', 'ED3049',
                 'PS312', 'LSJ1', 'JU258', 'MY16', 
                 'CB4852', 'CB4856', 'CB4853'
                 ]
reduced_strains = ['AQ1033', 'AQ1037', 'AQ1038', 'CB1069', 'CB5', 'ED3054', 'JU438',
         'MT2248', 'MT8504', 'N2', 'NL1137', 'RB2005', 'RB557', 'VC12']



def _h_get_paths(base_file):
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/classify_strains/results'
        main_file = os.path.join(os.environ['TMPDIR'], base_file)
        #main_file = '/work/ajaver/classify_strains/train_set/SWDB_skel_smoothed.hdf5'
    else:        
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains'
        main_file = os.path.join('/Users/ajaver/Desktop', base_file)
    
    return log_dir_root, main_file


sample_size_frames_s_dflt = 90.
sample_frequency_s_dflt = 1/10

def main(
    epochs = 5000,
    model_type = 'simple',
    is_reduced = False,
    is_wild_isolates = False,
    is_CeNDR = False,
    saving_period = None,
    model_path = None,
    is_angle = False,
    n_batch = None,
    sample_size_frames_s = sample_size_frames_s_dflt,
    sample_frequency_s = sample_frequency_s_dflt,
    is_QTL = 0
    ):

    
    # for reproducibility
    rand_seed = 1337
    np.random.seed(rand_seed)  
    output_activation = None
    
    if not is_CeNDR:
        base_file = 'SWDB_skel_smoothed.hdf5'
    else:
        base_file = 'CeNDR_skel_smoothed.hdf5'
    log_dir_root, main_file = _h_get_paths(base_file)
    
    if is_reduced:
      bn_prefix = 'R_'
      if is_CeNDR:
          valid_strains = CeNDR_base_strains
      else:
          valid_strains = reduced_strains
    elif is_wild_isolates:
      bn_prefix = 'W_'
      valid_strains = wild_isolates_WT2 
    else:
      bn_prefix = ''
      valid_strains = None
    print(valid_strains)

    if is_angle:
        bn_prefix += 'ang_'
    
    
    if saving_period is None:
      #the saving period must be larger in the reduce form since it has 
      #less samples per epoch
      if is_reduced or is_wild_isolates:
        saving_period = 12
      else:
        saving_period = 3
    
    loss = 'categorical_crossentropy'
    metrics = ['categorical_accuracy']
    
    if is_QTL == 1:
        loss = 'binary_crossentropy'
        metrics = ['binary_crossentropy']
        output_activation = 'sigmoid'
        bn_prefix += 'Q1_'
    elif is_QTL == 2:
        loss = 'mean_squared_error'
        metrics = ['mean_squared_error']
        output_activation = 'tanh'
        bn_prefix += 'Q2_'
    
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
        if 'FChead' in model_type:
            from models import FChead_resnet50_model
            model_fun = FChead_resnet50_model
        else:
            from models import resnet50_model
            if not '_D' in model_type:
              model_fun = resnet50_model
            else:
                dropout_rate = float(model_type.partition('_D')[-1])
                model_fun = partial(resnet50_model, dropout_rate=dropout_rate)
            
        
    else:
        ValueError('Not valid model_type')
    
    print(output_activation)
    if output_activation is not None:
        model_fun = partial(model_fun, output_activation=output_activation)
    
    if n_batch is None:
      #resnet use more memory i  have to reduce the batch size to fit it in the GPU    
      if 'resnet50' in model_type:
        n_batch_base = 32
      else:
        n_batch_base = 64

    factor = sample_size_frames_s/sample_size_frames_s_dflt
    factor *= (sample_frequency_s_dflt/sample_frequency_s)
    n_batch = max(int(math.floor(n_batch_base/factor)), 1)

    
    train_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = n_batch, 
                                   set_type='train',
                                   sample_size_frames_s = sample_size_frames_s,
                                   sample_frequency_s = sample_frequency_s,
                                   valid_strains = valid_strains,
                                   is_angle = is_angle,
                                   is_QTL = is_QTL
                                   )
    val_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = n_batch, 
                                   set_type='val',
                                   sample_size_frames_s = sample_size_frames_s,
                                   sample_frequency_s = sample_frequency_s,
                                   valid_strains = valid_strains,
                                   is_angle = is_angle,
                                   is_QTL = is_QTL
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
    

    print(model.summary())    
    print(model.get_layer('output').activation)
    print(train_generator.skeletons_indexes['strain'].unique())
    print(train_generator.n_batch)

    base_name = bn_prefix + model.name
    base_name = 'S{}_F{:.2}_{}'.format(sample_size_frames_s, sample_frequency_s, base_name)

    log_dir = os.path.join(log_dir_root, 'logsN', '%s_%s' % (base_name, time.strftime('%Y%m%d_%H%M%S')))
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
                  loss=loss,
                  metrics=metrics)
    
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