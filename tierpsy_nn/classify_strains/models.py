#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:55:36 2017

@author: ajaver
"""
import numpy as np

from keras.models import Model
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Dropout

#%%
def simple_model(input_shape, output_shape):
    img_input =  Input(shape = input_shape)
    
    x = Conv2D(32, (3, 3), padding='same', name='conv0')(img_input)
    x = Activation('relu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2), name='conv0_pool')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='conv1a')(x)
    x = BatchNormalization(name='conv1a_bn')(x)
    x = Activation('relu', name='conv1a_act')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='conv1b')(x)
    x = BatchNormalization(name='conv1b_bn')(x)
    x = Activation('relu', name='conv1b_act')(x)
    
    x = MaxPooling2D((2, 2), name='conv1_pool')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='conv2a')(x)
    x = BatchNormalization(name='conv2a_bn')(x)
    x = Activation('relu', name='conv2a_act')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='conv2b')(x)
    x = BatchNormalization(name='conv2b_bn')(x)
    x = Activation('relu', name='conv2b_act')(x)
    
    
    x = MaxPooling2D((2, 2), name='conv2_pool')(x)
    
    x = Conv2D(256, (3, 3), padding='same', name='conv3a')(x)
    x = BatchNormalization(name='conv3a_bn')(x)
    x = Activation('relu', name='conv3a_act')(x)
    
    x = Conv2D(256, (3, 3), padding='same', name='conv3b')(x)
    x = BatchNormalization(name='conv3b_bn')(x)
    x = Activation('relu', name='conv3b_act')(x)
    
    
    x = MaxPooling2D((2, 2), name='conv3_pool')(x)
    
    x = Conv2D(512, (3, 3), padding='same', name='conv4a')(x)
    x = BatchNormalization(name='conv4a_bn')(x)
    x = Activation('relu', name='conv4a_act')(x)
    
    x = Conv2D(512, (3, 3), padding='same', name='conv4b')(x)
    x = BatchNormalization(name='conv4b_bn')(x)
    x = Activation('relu', name='conv4b_act')(x)
    
    
    x = GlobalMaxPooling2D(name='avg_pool')(x)
    
    x = Dense(1024, name='dense0', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(np.prod(output_shape), name='dense1', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    output = Dense(np.prod(output_shape), name='output', activation='softmax')(x)
    output = Reshape(output_shape)(output)
    
    model = Model(img_input, output, name = 'simple_model')
    
    return model

#%%
import os
import time
import sys

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from skeletons_flow import SkeletonsFlow

if __name__ == '__main__':
    
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/classify_strains/results'
        main_file = '/work/ajaver/classify_strains/train_set/SWDB_skel_smoothed.hdf5'
    else:        
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains'
        main_file = '/Users/ajaver/Desktop/SWDB_skel_smoothed.hdf5'
    
    # for reproducibility
    rand_seed = 1337
    np.random.seed(rand_seed)  
    
    epochs = 500
    saving_period = 50
    
    skel_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = 50, 
                                   set_type='tiny'
                                   )
    X,Y = next(skel_generator)
    input_shape = X.shape[1:]
    output_shape = Y.shape[1:]
    
    model = simple_model(input_shape, output_shape)

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
    
    model.fit_generator(skel_generator,
                        steps_per_epoch = len(skel_generator), 
                        epochs = epochs,
                        verbose = 1,
                        callbacks=[tb, mcp]
                        )
