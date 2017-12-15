#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:13:12 2017

@author: ajaver
"""

import numpy as np

from keras import layers
from keras.layers import Input
from keras.layers import SeparableConv2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import BatchNormalization

from keras.layers.wrappers import TimeDistributed
    
from keras.models import Model

rand_seed = 1337
np.random.seed(rand_seed)  # for reproducibility
#%%
def _simple_model(input_shape, name='I_'):
    name += '_'
    
    input_data = Input(shape=input_shape, name=name+'input')
    
    x = Conv2D(32, (3, 3), padding='same', name=name+'conv0')(input_data)
    x = Activation('relu', name=name+'conv0_act')(x)
    x = MaxPooling2D((2, 2), name=name+'conv0_pool')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name=name+'conv1a')(x)
    x = BatchNormalization(name=name+'conv1a_bn')(x)
    x = Activation('relu', name=name+'conv1a_act')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name=name+'conv1b')(x)
    x = BatchNormalization(name=name+'conv1b_bn')(x)
    x = Activation('relu', name=name+'conv1b_act')(x)
    
    x = MaxPooling2D((2, 2), name=name+'conv1_pool')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name=name+'conv2a')(x)
    x = BatchNormalization(name=name+'conv2a_bn')(x)
    x = Activation('relu', name=name+'conv2a_act')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name=name+'conv2b')(x)
    x = BatchNormalization(name=name+'conv2b_bn')(x)
    x = Activation('relu', name=name+'conv2b_act')(x)
    
    
    x = MaxPooling2D((2, 2), name=name+'conv2_pool')(x)
    
    x = Conv2D(256, (3, 3), padding='same', name=name+'conv3a')(x)
    x = BatchNormalization(name=name+'conv3a_bn')(x)
    x = Activation('relu', name=name+'conv3a_act')(x)
    
    x = Conv2D(256, (3, 3), padding='same', name=name+'conv3b')(x)
    x = BatchNormalization(name=name+'conv3b_bn')(x)
    x = Activation('relu', name=name+'conv3b_act')(x)
    
    x = MaxPooling2D((2, 2), name=name+'conv3_pool')(x)
    
    x = Conv2D(512, (3, 3), padding='same', name=name+'conv4a')(x)
    x = BatchNormalization(name=name+'conv4a_bn')(x)
    x = Activation('relu', name=name+'conv4a_act')(x)
    
    x = Conv2D(512, (3, 3), padding='same', name=name+'conv4b')(x)
    x = BatchNormalization(name=name+'conv4b_bn')(x)
    x = Activation('relu', name=name+'conv4b_act')(x)
    
    output = GlobalMaxPooling2D(name=name+'avg_pool')(x)
    
    model = Model(input_data, output)
    return model

def simple_model(input_shape, output_shape):
    mod_s = _simple_model(input_shape)
    
    x = Dense(512, name='dense0', activation='elu')(mod_s.output)
    x = Dropout(0.4)(x)
    x = Dense(512, name='dense1', activation='elu')(x)
    x = Dropout(0.4)(x)
    x = Dense(output_shape[0]*output_shape[1])(x)
    x = Reshape(output_shape)(x)
    output = Activation('softmax')(x)
    model = Model(mod_s.input, output)
    
    return model
#%%
def model_timedistributed(roi_size, window_input, output_shape):
    
    input_d = Input(shape=(window_input, roi_size, roi_size, 1))
    
    mod1 = _simple_model((roi_size,roi_size,1))
    
    x = TimeDistributed(mod1)(input_d)
    x = Reshape((window_input*512,))(x)
    x = Dense(512, name='dense0', activation='elu')(x)
    x = Dropout(0.4)(x)
    x = Dense(512, name='dense1', activation='elu')(x)
    x = Dropout(0.4)(x)
    x = Dense(output_shape[0]*output_shape[1])(x)
    x = Reshape(output_shape)(x)
    output = Activation('softmax')(x)
    model = Model(input_d, output)
    
    return model
#%%
def model_separable(win_size, roi_size, nb_classes, y_offset=0):
    name = 'separable_conv2d'
    
    input_shape = (roi_size, roi_size, win_size)
    input_data = Input(shape=input_shape, name='input_all')
    
    x = Conv2D(32, (3, 3), padding='same', name=name+'conv0')(input_data)
    x = Activation('relu', name=name+'conv0_act')(x)
    x = MaxPooling2D((2, 2), name=name+'conv0_pool')(x)
    
    x = SeparableConv2D(64, (3, 3), use_bias=False, name=name+'conv1a')(x)
    x = BatchNormalization(name=name+'conv1a_bn')(x)
    x = Activation('relu', name=name+'conv1a_act')(x)
    
    x = SeparableConv2D(64, (3, 3), padding='same', name=name+'conv1b')(x)
    x = BatchNormalization(name=name+'conv1b_bn')(x)
    x = Activation('relu', name=name+'conv1b_act')(x)
    
    x = MaxPooling2D((2, 2), name=name+'conv1_pool')(x)
    
    x = SeparableConv2D(128, (3, 3), padding='same', name=name+'conv2a')(x)
    x = BatchNormalization(name=name+'conv2a_bn')(x)
    x = Activation('relu', name=name+'conv2a_act')(x)
    
    x = SeparableConv2D(128, (3, 3), padding='same', name=name+'conv2b')(x)
    x = BatchNormalization(name=name+'conv2b_bn')(x)
    x = Activation('relu', name=name+'conv2b_act')(x)
    
    
    x = MaxPooling2D((2, 2), name=name+'conv2_pool')(x)
    
    x = SeparableConv2D(256, (3, 3), padding='same', name=name+'conv3a')(x)
    x = BatchNormalization(name=name+'conv3a_bn')(x)
    x = Activation('relu', name=name+'conv3a_act')(x)
    
    x = SeparableConv2D(256, (3, 3), padding='same', name=name+'conv3b')(x)
    x = BatchNormalization(name=name+'conv3b_bn')(x)
    x = Activation('relu', name=name+'conv3b_act')(x)
    
    x = MaxPooling2D((2, 2), name=name+'conv3_pool')(x)
    
    x = SeparableConv2D(512, (3, 3), padding='same', name=name+'conv4a')(x)
    x = BatchNormalization(name=name+'conv4a_bn')(x)
    x = Activation('relu', name=name+'conv4a_act')(x)
    
    x = SeparableConv2D(512, (3, 3), padding='same', name=name+'conv4b')(x)
    x = BatchNormalization(name=name+'conv4b_bn')(x)
    x = Activation('relu', name=name+'conv4b_act')(x)
    
    x = GlobalMaxPooling2D(name=name+'avg_pool')(x)
    
    x = Dense(1024, name=name+'dense0', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(1024, name=name+'dense1', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense((win_size-y_offset)*nb_classes)(x)
    x = Reshape(((win_size-y_offset), nb_classes))(x)
    
    output = Activation('softmax')(x)
        
    model = Model(input_data, output)
    return model
#%%

if __name__ == '__main__':
    win_size = 5
    nb_classes = 2
    roi_size = 128
    y_offset_left = 2
    y_offset_right = 2

    input_shape1 = (roi_size, roi_size, win_size)
    input_shape2 = (win_size, roi_size, roi_size, 1)
    output_shape = (win_size-y_offset_left-y_offset_right, nb_classes)
    
    
    mod1 = simple_model(input_shape1, output_shape)
    mod2 = model_timedistributed(roi_size, win_size, output_shape)
    
    #mod = model_separable(win_size, roi_size, nb_classes, y_offset)