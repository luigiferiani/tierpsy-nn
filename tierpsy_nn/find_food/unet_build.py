#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:04:11 2017

@author: ajaver
"""

from math import ceil, floor

from keras.layers import Input 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Dropout 
from keras.layers import concatenate 
from keras.layers import Conv2DTranspose 
from keras.layers import Cropping2D 
from keras.layers import Activation 
from keras.layers import BatchNormalization 

from keras.models import Model

#%% METRICS
import tensorflow as tf
from keras.metrics import categorical_accuracy

_EPSILON = 10e-8    
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
    
def w_pix_categorical_crossentropy(w_y_true, y_pred, from_logits=False):
    """Categorical crossentropy between an output tensor 
    and a target tensor, using precalculated weights for each pixel.
    The weights should be in the last dimmension of the target array."""
    
    y_true = w_y_true[:,:,:,:-1]
    w_vec = w_y_true[:,:,:,-1][:,:,:,None]
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= tf.reduce_sum(y_pred,
                axis=len(y_pred.get_shape()) - 1,
                keep_dims=True)
    epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    return - tf.reduce_sum(w_vec * y_true * tf.log(y_pred),
                           axis=len(y_pred.get_shape()) - 1)
    
    
def w_categorical_accuracy(w_y_true, y_pred, from_logits=False):
    y_true = w_y_true[:,:,:,:-1]
    return categorical_accuracy(y_true, y_pred)

#%%
def get_unet_small(input_shape = (444, 444, 1), n_outputs=2):
    #    #NOTES:
    #    #Conv2D defaults are:
    #    #kernel_initializer='glorot_uniform' is also known as Xavier
    #    #padding='valid' means "no padding"
    #    
    #    # if we include the batch normalization we could use use_bias=False, 
    #    #x = BatchNormalization(name='norm_d0a-b')(x)
    
    
    data =  Input(shape=input_shape, name='loaddata')
    d0b = Conv2D(64, (3, 3), name='conv_d0a-b', activation='relu')(data)
    d0c = Conv2D(64, (3, 3), name='conv_d0b-c', activation='relu')(d0b)
    
    d1a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d0c-1a')(d0c)
    d1b = Conv2D(128, (3, 3), name='conv_d1a-b', activation='relu')(d1a)
    d1c = Conv2D(128, (3, 3), name='conv_d1b-c', activation='relu')(d1b)
    
    d2a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d1c-2a')(d1c)
    d2b = Conv2D(256, (3, 3), name='conv_d2a-b', activation='relu')(d2a)
    d2c = Conv2D(256, (3, 3), name='conv_d2b-c', activation='relu')(d2b)
    
    d3a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d2c-3a')(d2c)
    d3b = Conv2D(512, (3, 3), name='conv_d3a-b', activation='relu')(d3a)
    d3c = Conv2D(512, (3, 3), name='conv_d3b-c', activation='relu')(d3b)
    d3c = Dropout(0.5, name='dropout_d3c')(d3c)
    
    u2a = Conv2DTranspose(256, 
                          (2, 2), 
                          strides=(2, 2),
                          name='upconv_u3d_u2a', 
                          padding='valid',
                          activation='relu')(d3c)
    d2cc = Cropping2D(cropping=_get_crop_size(d3c, d2c), name= 'crop_d2c-d2cc')(d2c)
    u2b = concatenate([u2a, d2cc], axis=3,  name= 'concat_d2cc_u2a-b')
    u2c = Conv2D(256, (3, 3), name='conv_u2b-c', activation='relu')(u2b)
    u2d = Conv2D(256, (3, 3), name='conv_u2c-d', activation='relu')(u2c)
    
    u1a = Conv2DTranspose(128, 
                          (2, 2), 
                          strides=(2, 2),
                          name='upconv_u2d_u1a', 
                          padding='valid',
                          activation='relu')(u2d)
    d1cc = Cropping2D(cropping=_get_crop_size(u2d, d1c), name= 'crop_d1c-d1cc')(d1c)
    u1b = concatenate([u1a, d1cc], axis=3,  name= 'concat_d1cc_u1a-b')
    u1c = Conv2D(128, (3, 3), name='conv_u1b-c', activation='relu')(u1b)
    u1d = Conv2D(128, (3, 3), name='conv_u1c-d', activation='relu')(u1c)
    
    u0a = Conv2DTranspose(64, 
                          (2, 2), 
                          strides=(2, 2),
                          name='upconv_u1d_u0a', 
                          padding='valid',
                          activation='relu')(u1d)
    d0cc = Cropping2D(cropping=_get_crop_size(u1d, d0c), name= 'crop_d0c-d0cc')(d0c)
    u0b = concatenate([u0a, d0cc], axis=3,  name= 'concat_d0cc_u0a-b')
    u0c = Conv2D(64, (3, 3), name='conv_u0b-c', activation='relu')(u0b)
    u0d = Conv2D(64, (3, 3), name='conv_u0c-d', activation='relu')(u0c)
    
    score = Conv2D(n_outputs, (1, 1), name='conv_u0c-score', activation='relu')(u0d)
    loss = Activation('softmax')(score)
    
    model = Model(data, loss)
        
    return model

    
    
#%%

def get_unet_model(input_shape = (444, 444, 1), n_outputs=2):
    #    #NOTES:
    #    #Conv2D defaults are:
    #    #kernel_initializer='glorot_uniform' is also known as Xavier
    #    #padding='valid' means "no padding"
    #    
    #    # if we include the batch normalization we could use use_bias=False, 
    #    #x = BatchNormalization(name='norm_d0a-b')(x)
    
    
    data =  Input(shape=input_shape, name='loaddata')
    d0b = Conv2D(64, (3, 3), name='conv_d0a-b', activation='relu')(data)
    d0c = Conv2D(64, (3, 3), name='conv_d0b-c', activation='relu')(d0b)
    
    d1a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d0c-1a')(d0c)
    d1b = Conv2D(128, (3, 3), name='conv_d1a-b', activation='relu')(d1a)
    d1c = Conv2D(128, (3, 3), name='conv_d1b-c', activation='relu')(d1b)
    
    d2a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d1c-2a')(d1c)
    d2b = Conv2D(256, (3, 3), name='conv_d2a-b', activation='relu')(d2a)
    d2c = Conv2D(256, (3, 3), name='conv_d2b-c', activation='relu')(d2b)
    
    d3a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d2c-3a')(d2c)
    d3b = Conv2D(512, (3, 3), name='conv_d3a-b', activation='relu')(d3a)
    d3c = Conv2D(512, (3, 3), name='conv_d3b-c', activation='relu')(d3b)
    d3c = Dropout(0.5, name='dropout_d3c')(d3c)
    
    d4a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d3c-4a')(d3c)
    d4b = Conv2D(1024, (3, 3), name='conv_d4a-b', activation='relu')(d4a)
    d4c = Conv2D(1024, (3, 3), name='conv_d4b-c', activation='relu')(d4b)
    d4c = Dropout(0.5, name='dropout_d4c')(d4c)
    
    u3a = Conv2DTranspose(512, 
                          (2, 2), 
                          strides=(2, 2),
                          name='upconv_d4c_u3a', 
                          padding='valid',
                          activation='relu')(d4c)
    d3cc = Cropping2D(cropping=_get_crop_size(d4c, d3c), name= 'crop_d3c-d3cc')(d3c)
    u3b = concatenate([u3a, d3cc], axis=3,  name= 'concat_d3cc_u3a-b')
    u3c = Conv2D(512, (3, 3), name='conv_u3b-c', activation='relu')(u3b)
    u3d = Conv2D(512, (3, 3), name='conv_u3c-d', activation='relu')(u3c)
    
    u2a = Conv2DTranspose(256, 
                          (2, 2), 
                          strides=(2, 2),
                          name='upconv_u3d_u2a', 
                          padding='valid',
                          activation='relu')(u3d)
    d2cc = Cropping2D(cropping=_get_crop_size(u3d, d2c), name= 'crop_d2c-d2cc')(d2c)
    u2b = concatenate([u2a, d2cc], axis=3,  name= 'concat_d2cc_u2a-b')
    u2c = Conv2D(256, (3, 3), name='conv_u2b-c', activation='relu')(u2b)
    u2d = Conv2D(256, (3, 3), name='conv_u2c-d', activation='relu')(u2c)
    
    u1a = Conv2DTranspose(128, 
                          (2, 2), 
                          strides=(2, 2),
                          name='upconv_u2d_u1a', 
                          padding='valid',
                          activation='relu')(u2d)
    d1cc = Cropping2D(cropping=_get_crop_size(u2d, d1c), name= 'crop_d1c-d1cc')(d1c)
    u1b = concatenate([u1a, d1cc], axis=3,  name= 'concat_d1cc_u1a-b')
    u1c = Conv2D(128, (3, 3), name='conv_u1b-c', activation='relu')(u1b)
    u1d = Conv2D(128, (3, 3), name='conv_u1c-d', activation='relu')(u1c)
    
    u0a = Conv2DTranspose(64, 
                          (2, 2), 
                          strides=(2, 2),
                          name='upconv_u1d_u0a', 
                          padding='valid',
                          activation='relu')(u1d)
    d0cc = Cropping2D(cropping=_get_crop_size(u1d, d0c), name= 'crop_d0c-d0cc')(d0c)
    u0b = concatenate([u0a, d0cc], axis=3,  name= 'concat_d0cc_u0a-b')
    u0c = Conv2D(64, (3, 3), name='conv_u0b-c', activation='relu')(u0b)
    u0d = Conv2D(64, (3, 3), name='conv_u0c-d', activation='relu')(u0c)
    
    score = Conv2D(n_outputs, (1, 1), name='conv_u0c-score', activation='relu')(u0d)
    loss = Activation('softmax')(score)
    
    model = Model(data, loss)
        
    return model


def _get_crop_size(small_m, big_m):
    #Conv2DTranspose so i used the one before (it will become twice as large)
    #for some reason i cannot get the shape after 
    up_conv_s = 2 
    extra_d = [int(b)-up_conv_s*int(s) for s,b in zip(small_m.shape[1:3], big_m.shape[1:3])]
    crop_size = [( int(floor(x/2)), int(ceil(x/2)) ) for x in extra_d]
    return crop_size

BN_BIAS = False    
def conv2d_norm(top, n_filters, kernel_shape, layer_name):
    dd = Conv2D(n_filters, 
                kernel_shape, 
                use_bias=BN_BIAS, 
                name='conv_' + layer_name)(top)    
    
    dd = BatchNormalization(name='bn_' + layer_name)(dd)
    bottom = Activation('relu', name='relu_' + layer_name)(dd)
    return bottom

def deconv2d_norm(top, n_filters, kernel_shape, layer_name):
    dd = Conv2DTranspose(n_filters, 
                          kernel_shape, 
                          strides=(2, 2),
                          name='upconv_' + layer_name, 
                          padding='valid',
                          use_bias=BN_BIAS)(top)
    
    dd = BatchNormalization(name='bn_' + layer_name)(dd)
    bottom = Activation('relu', name='relu_' + layer_name)(dd)
    return bottom

def get_unet_model_bn(input_shape = (444, 444, 1), n_outputs=2):

    #    #NOTES:
    #    #Conv2D defaults are:
    #    #kernel_initializer='glorot_uniform' is also known as Xavier
    #    #padding='valid' means "no padding"
    #    
    #    # if we include the batch normalization we could use use_bias=False, 
    #    #x = BatchNormalization(name='norm_d0a-b')(x)
    
    data =  Input(shape=input_shape, name='loaddata')
    d0b = Conv2D(64, (3, 3), name='conv_d0a-b', activation='relu')(data)
    d0c = conv2d_norm(d0b, 64, (3, 3), layer_name='d0c')
    
    d1a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d0c-1a')(d0c)
    d1b = conv2d_norm(d1a, 128, (3, 3), layer_name='d1b')
    d1c = conv2d_norm(d1b, 128, (3, 3), layer_name='d1c')
    
    d2a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d1c-2a')(d1c)
    d2b = conv2d_norm(d2a, 256, (3, 3), layer_name='d2b')
    d2c = conv2d_norm(d2b, 256, (3, 3), layer_name='d2c')
    
    d3a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d2c-3a')(d2c)
    d3b = conv2d_norm(d3a, 512, (3, 3), layer_name='d3b')
    d3c = conv2d_norm(d3b, 512, (3, 3), layer_name='d3c')
    d3c = Dropout(0.5, name='dropout_d3c')(d3c)
    
    d4a = MaxPooling2D((2, 2), strides=(2, 2), name='pool_d3c-4a')(d3c)
    d4b = conv2d_norm(d4a, 1024, (3, 3), layer_name='d4b')
    d4c = conv2d_norm(d4b, 1024, (3, 3), layer_name='d4c')
    d4c = Dropout(0.5, name='dropout_d4c')(d4c)
    
    
    u3a = deconv2d_norm(d4c, 512, (2, 2), 'd4c_u3a')
    d3cc = Cropping2D(cropping=_get_crop_size(d4c, d3c), name= 'crop_d3c-d3cc')(d3c)
    u3b = concatenate([u3a, d3cc], axis=3,  name= 'concat_d3cc_u3a-b')
    u3c = conv2d_norm(u3b, 512, (3, 3), layer_name='u3b-c')
    u3d = conv2d_norm(u3c, 512, (3, 3), layer_name='u3c-d')
    
    u2a = deconv2d_norm(u3d, 256, (2, 2), 'u3d_u2a')
    d2cc = Cropping2D(cropping=_get_crop_size(u3d, d2c), name= 'crop_d2c-d2cc')(d2c)
    u2b = concatenate([u2a, d2cc], axis=3,  name= 'concat_d2cc_u2a-b')
    u2c = conv2d_norm(u2b, 256, (3, 3), layer_name='u2b-c')
    u2d = conv2d_norm(u2c, 256, (3, 3), layer_name='u2c-d')
    
    
    u1a = deconv2d_norm(u2d, 128, (2, 2), 'u2d_u1a')
    d1cc = Cropping2D(cropping=_get_crop_size(u2d, d1c), name= 'crop_d1c-d1cc')(d1c)
    u1b = concatenate([u1a, d1cc], axis=3,  name= 'concat_d1cc_u1a-b')
    u1c = conv2d_norm(u1b, 128, (3, 3), layer_name='u1b-c')
    u1d = conv2d_norm(u1c, 128, (3, 3), layer_name='u1c-d')
    
    u0a = deconv2d_norm(u1d, 64, (2, 2), 'u1d_u0a')
    d0cc = Cropping2D(cropping=_get_crop_size(u1d, d0c), name= 'crop_d0c-d0cc')(d0c)
    u0b = concatenate([u0a, d0cc], axis=3,  name= 'concat_d0cc_u0a-b')
    u0c = conv2d_norm(u0b, 64, (3, 3), layer_name='u0b-c')
    u0d = conv2d_norm(u0c, 64, (3, 3), layer_name='u0c-d')
    
    score = Conv2D(n_outputs, (1, 1), name='conv_u0c-score', activation='relu')(u0d)
    loss = Activation('softmax')(score)
    
    model = Model(data, loss)
        
    return model


if __name__ == '__main__':
    mod = get_unet_model((444, 444, 1))
    mod_bn = get_unet_small((260, 260, 1))