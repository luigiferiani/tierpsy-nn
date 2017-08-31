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

from keras.models import Sequential
from keras.models import Model

from keras import layers
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import UpSampling2D

from keras.optimizers import Adam

def test_res():
    input_shape = (roi_size, roi_size, 1)
    img_input =  Input(shape=input_shape)
    x = Conv2D(32, (3, 3), use_bias=False, name='block1_conv1')(img_input) #strides=(2, 2),
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)
    
    
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name='block2_conv1')(x)
    x = BatchNormalization(name='block2_conv1_bn')(x)
    x = Activation('relu', name='block2_conv2_act')(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name='block2_conv2')(x)
    x = BatchNormalization(name='block2_conv2_bn')(x)
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])
    
    
    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    
    
    x = Activation('relu', name='block3_conv1_act')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, name='block3_conv1')(x)
    x = BatchNormalization(name='block3_conv1_bn')(x)
    x = Activation('relu', name='block3_conv2_act')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, name='block3_conv2')(x)
    x = BatchNormalization(name='block3_conv2_bn')(x)
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])
    
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(out_size[0]*out_size[1], activation='relu', name='skeleton')(x)
    x = Reshape((out_size))(x)
    
    model = Model(img_input, x, name='test4_xception')
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=1e-3, decay=0.05),
                  metrics=['mean_absolute_percentage_error'])


def test_large():
    #does not converge
    x = Conv2D(16, (3, 3), use_bias=False, name='conv0')(img_input)
    x = BatchNormalization(name='conv0_bn')(x)
    x = Activation('relu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2), padding='same', name='conv0_pool')(x)
    
    x = Conv2D(32, (3, 3), use_bias=False, name='conv1_1')(img_input)
    x = BatchNormalization(name='conv1_bn1')(x)
    x = Activation('relu', name='conv1_act1')(x)
    
    x = Conv2D(32, (3, 3), use_bias=False, name='conv1_2')(img_input)
    x = BatchNormalization(name='conv1_bn2')(x)
    x = Activation('relu', name='conv1_act2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='conv1_pool')(x)
    
    x = Conv2D(64, (3, 3), use_bias=False, name='conv2_1')(x)
    x = BatchNormalization(name='conv2_bn1')(x)
    x = Activation('relu', name='conv2_act1')(x)
    
    x = Conv2D(64, (3, 3), use_bias=False, name='conv2_2')(x)
    x = BatchNormalization(name='conv2_bn2')(x)
    x = Activation('relu', name='conv2_act2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='conv2_pool')(x)
    
    x = Conv2D(128, (3, 3), use_bias=False, name='conv3_1')(x)
    x = BatchNormalization(name='conv3_bn1')(x)
    x = Activation('relu', name='conv3_act1')(x)
    
    x = Conv2D(128, (3, 3), use_bias=False, name='conv3_2')(x)
    x = BatchNormalization(name='conv3_bn2')(x)
    x = Activation('relu', name='conv3_act2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='conv3_pool')(x)
    
    x = Flatten()(x)
    x = Dense(4092, activation='relu', name='dense')(x)
    x = Dense(out_size[0]*out_size[1], activation='tanh', name='skeleton')(x)
    x = Reshape((out_size))(x)

def test4():
    #does not converge
    x = Conv2D(32, (3, 3), name='conv0_1')(img_input)
    x = BatchNormalization(name='conv0_bn1')(x)
    x = Activation('relu', name='conv0_act1')(x)
    
    #x = Conv2D(32, (3, 3), name='conv0_2')(img_input)
    #x = BatchNormalization(name='conv0_bn2')(x)
    #x = Activation('relu', name='conv0_act2')(x)
    
    x = MaxPooling2D((2, 2), name='conv0_pool')(x)
    
    x = Conv2D(64, (3, 3), name='conv1_1')(x)
    x = BatchNormalization(name='conv1_bn1')(x)
    x = Activation('relu', name='conv1_act1')(x)
    x = MaxPooling2D((2, 2), name='conv1_pool')(x)
    
    x = Conv2D(128, (3, 3), name='conv2_1')(x)
    x = BatchNormalization(name='conv2_bn1')(x)
    x = Activation('relu', name='conv2_act1')(x)
    
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='dense1')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1024, activation='relu', name='dense2')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256, activation='relu', name='dense3')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(out_size[0]*out_size[1], activation='relu', name='skeleton')(x)
    x = Reshape((out_size))(x)


def test5():
    #converges based in the best results i have gotten so far
#    x = Conv2D(64, (3, 3), name='conv0')(img_input)
#    x = Activation('relu', name='conv0_act')(x)
#    
#    x = Conv2D(64, (3, 3), name='conv1')(img_input)
#    x = Activation('relu', name='conv1_act')(x)
#    x = MaxPooling2D((2, 2), name='conv1_pool')(x)
#    
    x = Conv2D(64, (3, 3), name='conv2')(img_input)
    x = Activation('relu', name='conv2_act')(x)
    x = MaxPooling2D((2, 2), name='conv2_pool')(x)
    
    x = Flatten()(x)
    
    x = Dense(1024, name='dense1', activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, name='dense2', activation='relu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, name='dense3', activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(out_size[0]*out_size[1], activation='relu', name='skeleton')(x)
    x = Reshape((out_size))(x)

def test6():
    #very slow, maybe stuck at 20
    input_shape = (roi_size, roi_size, 1)
    img_input =  Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), name='conv0')(img_input)
    x = Activation('relu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2), name='conv0_pool')(x)
    
    x = Conv2D(128, (3, 3), name='conv1')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_act')(x)
    x = MaxPooling2D((2, 2), name='conv1_pool')(x)
    
    x = Conv2D(256, (3, 3), name='conv2')(x)
    x = BatchNormalization(name='conv2_bn')(x)
    x = Activation('relu', name='conv2_act')(x)
    x = MaxPooling2D((2, 2), name='conv2_pool')(x)
    
    
    x = Flatten()(x)
    
    x = Dense(1024, name='dense1', activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, name='dense2', activation='relu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, name='dense3', activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(out_size[0]*out_size[1], activation='relu', name='skeleton')(x)
    x = Reshape((out_size))(x)

def test7():
    #does not converge
    
    
    x = Conv2D(64, (3, 3), name='conv0')(img_input)
    x = Activation('relu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2), name='conv0_pool')(x)
    
    x = Conv2D(128, (3, 3), name='conv1')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_act')(x)
    x = MaxPooling2D((2, 2), name='conv1_pool')(x)
    
    x = Conv2D(256, (3, 3), name='conv2')(x)
    x = BatchNormalization(name='conv2_bn')(x)
    x = Activation('relu', name='conv2_act')(x)
    x = MaxPooling2D((2, 2), name='conv2_pool')(x)
    
    x = Conv2D(512, (3, 3), name='conv3')(x)
    x = BatchNormalization(name='conv3_bn')(x)
    x = Activation('relu', name='conv3_act')(x)
    x = MaxPooling2D((2, 2), name='conv3_pool')(x)
    
    x = Conv2D(1024, (3, 3), name='conv4')(x)
    x = BatchNormalization(name='conv4_bn')(x)
    x = Activation('relu', name='conv4_act')(x)
    
    x = Flatten()(x)
    #x = GlobalAveragePooling2D(name='avg_pool')(x)
    
    x = Dense(1024, name='dense1', activation='tanh')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, name='dense2', activation='tanh')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, name='dense3', activation='tanh')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(out_size[0]*out_size[1], activation='tanh', name='skeleton')(x)
    x = Reshape((out_size))(x)
    
    
    model = Model(img_input, x, name='test4_xception')
    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(lr=1e-3),#, decay=0.05),
                  metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])

def test8():
#kinda works, but the output get stucks in the lower, cannot make it convert properly
#i suspect it is related with the padding
    x = Conv2D(32, (3, 3), padding='same', name='conv0')(img_input)
    x = Activation('elu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2), name='conv0_pool')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='conv1a')(x)
    x = BatchNormalization(name='conv1a_bn')(x)
    x = Activation('elu', name='conv1a_act')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='conv1b')(x)
    x = BatchNormalization(name='conv1b_bn')(x)
    x = Activation('elu', name='conv1b_act')(x)
    x = MaxPooling2D((2, 2), name='conv1_pool')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='conv2a')(x)
    x = BatchNormalization(name='conv2a_bn')(x)
    x = Activation('elu', name='conv2a_act')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='conv2b')(x)
    x = Activation('elu', name='conv2b_act')(x)
    x = GlobalMaxPooling2D(name='avg_pool')(x)
    
    x = Dense(128, name='dense1', activation='elu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, name='dense2', activation='elu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(out_size[0]*out_size[1], activation='elu', name='skeleton')(x)
    x = Reshape((out_size))(x)


def test9():
    x = Conv2D(32, (3, 3), padding='same', name='conv0')(img_input)
    x = Activation('elu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2), name='conv0_pool')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='conv1a')(x)
    x = BatchNormalization(name='conv1a_bn')(x)
    x = Activation('elu', name='conv1a_act')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='conv1b')(x)
    x = BatchNormalization(name='conv1b_bn')(x)
    x = Activation('elu', name='conv1b_act')(x)
    
    x = MaxPooling2D((2, 2), name='conv1_pool')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='conv2a')(x)
    x = BatchNormalization(name='conv2a_bn')(x)
    x = Activation('elu', name='conv2a_act')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='conv2b')(x)
    x = BatchNormalization(name='conv2b_bn')(x)
    x = Activation('elu', name='conv2b_act')(x)
    
    
    #x = GlobalMaxPooling2D(name='avg_pool')(x)
    
    x = Flatten()(x)
    x = Dense(2048, name='dense0', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(1024, name='dense1', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, name='dense2', activation='elu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(out_size[0]*out_size[1], activation='elu', name='skeleton')(x)
    x = Reshape((out_size))(x)

    model = Model(img_input, x)
    optimizer = Adam(lr=1e-3)#, decay=0.05)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])

def test10():
    '''This one works, but it has problems with the coils, and do not match the data 100%
    After a few iterations if fails to converge, but find things
    '''
    
    input_shape = (roi_size, roi_size, 1)
    img_input =  Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), padding='same', name='conv0')(img_input)
    x = Activation('elu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2), name='conv0_pool')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='conv1a')(x)
    x = BatchNormalization(name='conv1a_bn')(x)
    x = Activation('elu', name='conv1a_act')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='conv1b')(x)
    x = BatchNormalization(name='conv1b_bn')(x)
    x = Activation('elu', name='conv1b_act')(x)
    
    x = MaxPooling2D((2, 2), name='conv1_pool')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='conv2a')(x)
    x = BatchNormalization(name='conv2a_bn')(x)
    x = Activation('elu', name='conv2a_act')(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='conv2b')(x)
    x = BatchNormalization(name='conv2b_bn')(x)
    x = Activation('elu', name='conv2b_act')(x)
    
    
    x = MaxPooling2D((2, 2), name='conv2_pool')(x)
    
    x = Conv2D(256, (3, 3), padding='same', name='conv3a')(x)
    x = BatchNormalization(name='conv3a_bn')(x)
    x = Activation('elu', name='conv3a_act')(x)
    
    x = Conv2D(256, (3, 3), padding='same', name='conv3b')(x)
    x = BatchNormalization(name='conv3b_bn')(x)
    x = Activation('elu', name='conv3b_act')(x)
    
    
    x = GlobalMaxPooling2D(name='avg_pool')(x)
    
    #x = Flatten()(x)
    x = Dense(1024, name='dense0', activation='elu')(x)
    #x = Dropout(0.4)(x)
    
    x = Dense(512, name='dense1', activation='elu')(x)
    #x = Dropout(0.4)(x)
    
    x = Dense(128, name='dense2', activation='elu')(x)
    #x = Dropout(0.2)(x)
    
    x = Dense(out_size[0]*out_size[1], activation='elu', name='skeleton')(x)
    x = Reshape((out_size))(x)
    
    
    
    model = Model(img_input, x)
    optimizer = Adam(lr=1e-3)#, decay=0.05)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])

#%%
def test_resnet(out_size = (49, 2)):
    # for reproducibility
    rand_seed = 1337
    np.random.seed(rand_seed)  
    
    
    roi_size = 128
    
    
   
    input_shape = (roi_size, roi_size, 1)
    img_input =  Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), padding='same', use_bias=False, name='conv0')(img_input)
    x = BatchNormalization(name='conv0a_bn')(x)
    x = Activation('relu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2), name='conv0_pool')(x)
    
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv1a')(x)
    x = BatchNormalization(name='conv1a_bn')(x)
    x = Activation('relu', name='conv1a_act')(x)
    
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)
    x = MaxPooling2D((2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])
    
    
    residual = Conv2D(512, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = Conv2D(512, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)
    x = MaxPooling2D((2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])
    
    
    x = Conv2D(1024, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)
    
    x = GlobalMaxPooling2D(name='avg_pool')(x)
    
    
    #layers seems to be key to capture the worm shape
    #i need to use ELU instead of RELU otherwise the skeletons converges to 0
    x = Dense(1024, name='dense0', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(512, name='dense1', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, name='dense2', activation='elu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, name='dense3', activation='elu')(x)
    x = Dropout(0.1)(x)
    
    x = Dense(out_size[0]*out_size[1], activation='elu', name='skeleton')(x)
    x = Reshape((out_size))(x)
    
    
    
    model = Model(img_input, x)
    optimizer = Adam(lr=1e-3)#, decay=0.05)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])
    
def test_pyramid(out_size = (49, 2)):

    rand_seed = 1337
    np.random.seed(rand_seed)  
    
    
    roi_size = 128
    
    input_shape = (roi_size, roi_size, 1)
    img_input =  Input(shape=input_shape)
    
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
    x = MaxPooling2D((2, 2), name='conv2_pool')(x)
    
    x = Conv2D(256, (3, 3), padding='same', name='conv2b')(x)
    x = BatchNormalization(name='conv2b_bn')(x)
    x = Activation('relu', name='conv2b_act')(x)
    
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2,2))(x)
    
    x = Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2,2))(x)
    
    x = Conv2D(1, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(1024, name='dense0', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, name='dense1', activation='elu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(out_size[0]*out_size[1], activation='elu', name='skeleton')(x)
    x = Reshape((out_size))(x)

#%%
# for reproducibility
def test_pyramid_feat(out_size = (49, 2)):
    
    roi_size = 128
    rand_seed = 1337
    np.random.seed(rand_seed)
     
    input_shape = (roi_size, roi_size, 1)
    img_input =  Input(shape=input_shape)
    
    block1 = Conv2D(32, (3, 3), padding='same', name='conv0a')(img_input)
    block1 = Activation('relu', name='conv0a_act')(block1)
    block1 = Conv2D(32, (3, 3), padding='same', name='conv0b')(block1)
    block1 = Activation('relu', name='conv0b_act')(block1)
    
    block2 = MaxPooling2D((2, 2), name='conv0_pool')(block1)
    block2 = Conv2D(64, (3, 3), padding='same', name='conv1a')(block2)
    block2 = BatchNormalization(name='conv1a_bn')(block2)
    block2 = Activation('relu', name='conv1a_act')(block2)
    block2 = Conv2D(64, (3, 3), padding='same', name='conv1b')(block2)
    block2 = BatchNormalization(name='conv1b_bn')(block2)
    block2 = Activation('relu', name='conv1b_act')(block2)
    
    block3 = MaxPooling2D((2, 2), name='conv1_pool')(block2)
    block3 = Conv2D(128, (3, 3), padding='same', name='conv2a')(block3)
    block3 = BatchNormalization(name='conv2a_bn')(block3)
    block3 = Activation('relu', name='conv2a_act')(block3)
    block3 = Conv2D(128, (3, 3), padding='same', name='conv2b')(block3)
    block3 = BatchNormalization(name='conv2b_bn')(block3)
    block3 = Activation('relu', name='conv2b_act')(block3)
    
    block4 = MaxPooling2D((2, 2), name='conv2_pool')(block3)
    block4 = Conv2D(256, (3, 3), padding='same', name='conv3a')(block4)
    block4 = BatchNormalization(name='conv3a_bn')(block4)
    block4 = Activation('relu', name='conv3a_act')(block4)
    block4 = Conv2D(256, (3, 3), padding='same', name='conv3b')(block4)
    block4 = BatchNormalization(name='conv3b_bn')(block4)
    block4 = Activation('relu', name='conv3b_act')(block4)
    
    feat_top = GlobalMaxPooling2D(name='avg_pool')(block4)
    feat_top = Dense(1024, name='dense0', activation='elu')(feat_top)
    feat_top = Dropout(0.4)(feat_top)
    
    
    down_block4 = Conv2D(128, (3, 3), padding='same')(block4)
    down_block4 = UpSampling2D((2,2))(down_block4)
    p_block4 = layers.add([block3, down_block4])
    feat_p4 = GlobalMaxPooling2D(name='avg_pool_4')(p_block4)
    feat_p4 = Dense(1024, activation='elu')(feat_p4)
    feat_p4 = Dropout(0.4)(feat_p4)
    
    down_block3 = Conv2D(64, (3, 3), padding='same')(block3)
    down_block3 = UpSampling2D((2,2))(down_block3)
    p_block3 = layers.add([block2, down_block3])
    feat_p3 = GlobalMaxPooling2D(name='avg_pool_3')(p_block3)
    
    
    
    feat_p3 = Dense(1024, activation='elu')(feat_p3)
    feat_p3 = Dropout(0.4)(feat_p3)
    
    #down_merge_3_4 = Conv2D(32, (1, 1), activation='elu', padding='same')(merged_3_4)
    #down_merge_3_4 = UpSampling2D((2,2))(down_merge_3_4)
    #lat_block2 = Conv2D(32, (1, 1), activation='elu', padding='same')(block2)
    #merged_2_3 = layers.add([lat_block2, down_merge_3_4])
    #
    #feat_2_3 = Conv2D(16, (3, 3), activation='elu', padding='same')(merged_2_3)
    #feat_2_3 = Conv2D(1, (3, 3), activation='elu', padding='same')(feat_2_3)
    #feat_2_3 = Flatten()(feat_2_3)
    #feat_2_3 = Dense(1024, activation='elu')(feat_2_3)
    #
    #down_merge_2_3 = Conv2D(16, (1, 1), activation='elu', padding='same')(merged_2_3)
    #down_merge_2_3 = UpSampling2D((2,2))(down_merge_2_3)
    #lat_block1 = Conv2D(16, (1, 1), activation='elu', padding='same')(block1)
    #merged_1_2 = layers.add([lat_block1, down_merge_2_3])
    #
    #feat_1_2 = Conv2D(1, (3, 3), activation='elu', padding='same')(merged_1_2)
    #feat_1_2 = Flatten()(feat_1_2)
    #feat_1_2 = Dense(1024, activation='elu')(feat_1_2)
    
    #all_feats = layers.add([feat_top, feat_3_4, feat_2_3, feat_1_2])
    all_feats = layers.add([feat_top, feat_p4, feat_p3])
    all_feats = Dense(1024, activation='elu')(all_feats)
    p_block4 = Dense(1024, activation='elu')(p_block4)
    p_block4 = Dropout(0.4)(feat_top)
    p_block4 = Dense(1024, activation='elu')(p_block4)
    p_block4 = Dropout(0.4)(feat_top)
    
    
    
    skel_out = Dense(out_size[0]*out_size[1], activation='elu', name='skeleton')(all_feats)
    skel_out = Reshape((out_size))(skel_out)
    
    
    
    model = Model(img_input, skel_out)
    #optimizer = Adam(lr=1e-2, decay=0.1)
    optimizer = Adam(lr=1e-3, decay=0.1)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])

    return model
#%%
# for reproducibility
def test_pyramid_feat2(out_size = (49, 2)):
    
    roi_size = 128
    rand_seed = 1337
    np.random.seed(rand_seed)
     
    input_shape = (roi_size, roi_size, 1)
    img_input =  Input(shape=input_shape)
    
    block1 = Conv2D(32, (3, 3), padding='same', name='conv0a')(img_input)
    block1 = Activation('relu', name='conv0a_act')(block1)
    block1 = Conv2D(32, (3, 3), padding='same', name='conv0b')(block1)
    block1 = Activation('relu', name='conv0b_act')(block1)
    
    block2 = MaxPooling2D((2, 2), name='conv0_pool')(block1)
    block2 = Conv2D(64, (3, 3), padding='same', name='conv1a')(block2)
    block2 = BatchNormalization(name='conv1a_bn')(block2)
    block2 = Activation('relu', name='conv1a_act')(block2)
    block2 = Conv2D(64, (3, 3), padding='same', name='conv1b')(block2)
    block2 = BatchNormalization(name='conv1b_bn')(block2)
    block2 = Activation('relu', name='conv1b_act')(block2)
    
    block3 = MaxPooling2D((2, 2), name='conv1_pool')(block2)
    block3 = Conv2D(128, (3, 3), padding='same', name='conv2a')(block3)
    block3 = BatchNormalization(name='conv2a_bn')(block3)
    block3 = Activation('relu', name='conv2a_act')(block3)
    block3 = Conv2D(128, (3, 3), padding='same', name='conv2b')(block3)
    block3 = BatchNormalization(name='conv2b_bn')(block3)
    block3 = Activation('relu', name='conv2b_act')(block3)
    
    block4 = MaxPooling2D((2, 2), name='conv2_pool')(block3)
    block4 = Conv2D(256, (3, 3), padding='same', name='conv3a')(block4)
    block4 = BatchNormalization(name='conv3a_bn')(block4)
    block4 = Activation('relu', name='conv3a_act')(block4)
    block4 = Conv2D(256, (3, 3), padding='same', name='conv3b')(block4)
    block4 = BatchNormalization(name='conv3b_bn')(block4)
    block4 = Activation('relu', name='conv3b_act')(block4)
    
    feat_top = GlobalMaxPooling2D(name='avg_pool')(block4)
    feat_top = Dense(1024, name='dense0', activation='elu')(feat_top)
    feat_top = Dropout(0.4)(feat_top)
    
    down_block4 = Conv2D(1, (1, 1), padding='same')(block4)
    down_block4 = UpSampling2D((2,2))(down_block4)
    lat_block3 = Conv2D(1, (1, 1), padding='same')(block3)
    p_block3 = layers.add([lat_block3, down_block4])
    
    down_block3 = UpSampling2D((2,2))(p_block3)
    lat_block2 = Conv2D(1, (1, 1), padding='same')(block2)
    p_block2 = layers.add([lat_block2, down_block3])
    
    
    feat_bot = Flatten()(p_block2)
    feat_bot = Dense(1024, activation='elu')(feat_bot)
    feat_bot = Dropout(0.4)(feat_bot)
    
    all_feats = layers.add([feat_top, feat_bot])
    all_feats = Dense(1024, activation='elu')(all_feats)
    feat_bot = Dropout(0.4)(feat_bot)
    
    skel_out = Dense(out_size[0]*out_size[1], activation='elu', name='skeleton')(all_feats)
    skel_out = Reshape((out_size))(skel_out)
    
    
    
    model = Model(img_input, skel_out)
    #optimizer = Adam(lr=1e-2, decay=0.1)
    optimizer = Adam(lr=1e-3, decay=0.1)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])

    return model, 'pyramid_feat2'
#%%
def test_simple_v2(out_size = (49, 2), roi_size = 128, lr=1e-3, decay=0.05):
    # for reproducibility
    rand_seed = 1337
    np.random.seed(rand_seed)  
    
    input_shape = (roi_size, roi_size, 1)
    img_input =  Input(shape=input_shape)
    
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
    
    x = Dense(1024, name='dense1', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(np.prod(out_size), activation='elu', name='skeleton')(x)
    x = Reshape((out_size))(x)
    
    
    
    model = Model(img_input, x)
    optimizer = Adam(lr=1e-3, decay=0.05)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])

    model_name = 'simple_v2'
    return model, model_name

#%%

def test_simple_head(out_size = (49, 2), roi_size = 128, lr=1e-3, decay=0.05):
    # for reproducibility
    rand_seed = 1337
    np.random.seed(rand_seed)  
    
    input_shape = (roi_size, roi_size, 1)
    img_input =  Input(shape=input_shape)
    
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
    
    x = Dense(512, name='dense0', activation='tanh')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, name='dense1', activation='tanh')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(np.prod(out_size), activation='elu', name='skeleton')(x)
    x = Reshape((out_size))(x)
    
    
    
    model = Model(img_input, x)
    optimizer = Adam(lr=1e-4, decay=0.05)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])

    model_name = 'test_simple_head'
    return model, model_name
