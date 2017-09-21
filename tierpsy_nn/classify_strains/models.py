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
from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras import layers

from keras import backend as K
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

def larger_model(input_shape, output_shape):
    img_input =  Input(shape = input_shape)
    
    #there is a bug here (the kernel must be 3x7) but i keep it just to document it
    x = Conv2D(32, (3, 7), padding='same', name='conv0')(img_input)
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
    
    x = MaxPooling2D((2, 2), name='conv4_pool')(x)
    
    x = Conv2D(1024, (3, 3), padding='same', name='conv5a')(x)
    x = BatchNormalization(name='conv5a_bn')(x)
    x = Activation('relu', name='conv5a_act')(x)
    
    x = Conv2D(1024, (3, 3), padding='same', name='conv5b')(x)
    x = BatchNormalization(name='conv5b_bn')(x)
    x = Activation('relu', name='conv5b_act')(x)
    
    x = GlobalMaxPooling2D(name='avg_pool')(x)
    x = Dense(1024, name='dense0', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    output = Dense(np.prod(output_shape), name='output', activation='softmax')(x)
    output = Reshape(output_shape)(output)
    
    model = Model(img_input, output, name = 'larger_model')
    
    return model

#%% Modified from https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
def _identity_block(input_tensor, kernel_size, filters, stage, block, dropout_rate=0):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    '''
     #This is a bottleneck architecture 1x1 conv + 3x3 conv + 1x1 conv
     
     used to reduce the number of computation between layers
    '''
    
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


   
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
    
    if dropout_rate > 0:
        #in the wide resnet they use dropout. I leave it in case it becomes necessary
        x = Dropout(dropout_rate)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def _conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dropout_rate=dropout_rate):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    
    '''
    This block is used to reduce the the size of the features maps using strides instead of maxpool
    '''
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    if dropout_rate > 0:
        #in the wide resnet they use dropout. I leave it in case it becomes necessary
        x = Dropout(dropout_rate)(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def resnet50_model(input_shape, output_shape, dropout_rate=0.0):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    img_input =  Input(shape = input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [64, 64, 256], stage=2, block='b', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [64, 64, 256], stage=2, block='c', dropout_rate=dropout_rate)

    x = _conv_block(x, 3, [128, 128, 512], stage=3, block='a', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block='b', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block='c', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block='d', dropout_rate=dropout_rate)

    x = _conv_block(x, 3, [256, 256, 1024], stage=4, block='a', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dropout_rate=dropout_rate)

    x = _conv_block(x, 3, [512, 512, 2048], stage=5, block='a', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dropout_rate=dropout_rate)
    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dropout_rate=dropout_rate)

    x = AveragePooling2D((7, 2), name='avg_pool')(x)
    #x = Flatten()(x)
    x = GlobalMaxPooling2D()(x)
    output = Dense(np.prod(output_shape), name='output', activation='softmax')(x)
    output = Reshape(output_shape)(output)
    
    model = Model(img_input, output, name = 'resnet50_D{}'.format(float(dropout_rate)))
    
    return model
#%%
##%%
def main():
    #%%
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
    
    # for reproducibility
    rand_seed = 1337
    np.random.seed(rand_seed)  
    
    epochs = 500
    saving_period = 64
    
    skel_generator = SkeletonsFlow(main_file = main_file, 
                                   n_batch = 32, 
                                   set_type='tiny'
                                   )
    X,Y = next(skel_generator)
    input_shape = X.shape[1:]
    output_shape = Y.shape[1:]
    #%%
    model = resnet50_model(input_shape, output_shape, dropout_rate=0.2)

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

if __name__ == '__main__':
    input_shape = (900, 49, 2)
    output_shape = (356,)
    
    main()
    #mod = resnet50_model(input_shape, output_shape)
    #mod = larger_model(input_shape, output_shape)