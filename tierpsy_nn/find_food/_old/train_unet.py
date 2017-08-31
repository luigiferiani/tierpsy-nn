#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:28:34 2017

@author: ajaver
"""
import os
import time
import numpy as np

from tensorflow.contrib import keras
TensorBoard = keras.callbacks.TensorBoard
ModelCheckpoint = keras.callbacks.ModelCheckpoint
K = keras.backend
Adam = keras.optimizers.Adam

#from keras.callbacks import TensorBoard, ModelCheckpoint
#from keras import backend as K
#from keras.optimizers import Adam


from augmentation import get_sizes, ImageMaskGenerator, DirectoryImgGenerator
from unet_build import get_unet_model

##from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

if __name__ == '__main__':
    main_dir = '/Users/ajaver/OneDrive - Imperial College London/food/train_set'
    SAVE_DIR = '/Users/ajaver/OneDrive - Imperial College London/food/results'
    model_name = 'unet'
    
    epochs = 100
    batch_size = 128
    im_size = (512, 512)
    transform_ags = dict(rotation_range=90, 
         shift_range = 0.1,
         horizontal_flip=True,
         vertical_flip=True,
         alpha_range=200,
         sigma=10)
    
    log_dir = os.path.join(SAVE_DIR, 'logs', '%s_%s' % (model_name, time.strftime('%Y%m%d_%H%M%S')))
    pad=int(np.ceil(np.log10(epochs+1)))
    checkpoint_file = os.path.join(log_dir, '%s-{epoch:0%id}-{loss:.4f}.h5' % (model_name, pad))
    tb = TensorBoard(log_dir=log_dir)
    mcp = ModelCheckpoint(checkpoint_file, verbose=1,  mode='auto', period=1)
    
    input_size, output_size, pad_size, tile_corners = get_sizes(im_size)
    
    bs = int(round(batch_size/len(tile_corners)))
    img_generator = ImageMaskGenerator(DirectoryImgGenerator(main_dir), 
                             transform_ags, 
                             pad_size,
                             input_size,
                             tile_corners,
                             batch_size=bs,
                             epoch_size = bs*20,
                             )
    
    model = get_unet_model()
    model.compile(optimizer=Adam(lr=1e-5), 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_crossentropy'])
    
    model.fit_generator(img_generator,
                        steps_per_epoch = round(img_generator.tot_samples/batch_size), 
                        epochs = epochs,
                        verbose = 1,
                        callbacks=[tb, mcp]) #some how they were breaking the mpc ? this crashes...