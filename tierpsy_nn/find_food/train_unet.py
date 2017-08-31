#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:28:34 2017

@author: ajaver
"""
import os
import time
import numpy as np

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers import RMSprop

from augmentation import get_sizes, ImageMaskGenerator, DirectoryImgGenerator
from unet_build import get_unet_model

main_dir = '/work/ajaver/food/train_set'
SAVE_DIR = '/work/ajaver/food/results'

#main_dir = '/Users/ajaver/OneDrive - Imperial College London/food/train_set'
#SAVE_DIR = '/Users/ajaver/OneDrive - Imperial College London/food/results'

def main(
    epochs = 20000,
    batch_size = 6,
    saving_period = 250,
    im_size = (512, 512),
    only_contours = False,
    optimizer = 'RMSprop',
    lr = 1e-5
    ):
    
    model = get_unet_model()
    model_name = 'unet_{}{}'.format(optimizer, int(np.log10(lr)))
    
    if optimizer == 'Adam':
        optimizer = Adam(lr=lr)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(lr=lr, rho=0.99)
    
    if only_contours:
        model_name += '_cnt'

    n_tiles = batch_size
    transform_ags = dict(
            rotation_range=90, 
             shift_range = 0.1,
             zoom_range = (0.9, 1.5),
             horizontal_flip=True,
             vertical_flip=True,
             elastic_alpha_range=400,
             elastic_sigma=20,
             int_alpha=(0.5,2.25)
             )
        
    weight_params = dict(
            sigma=2.5,
            weigth=20
            )
    
    log_dir = os.path.join(SAVE_DIR, 'logs', '%s_%s' % (model_name, time.strftime('%Y%m%d_%H%M%S')))
    pad=int(np.ceil(np.log10(epochs+1)))
    checkpoint_file = os.path.join(log_dir, '%s-{epoch:0%id}-{loss:.4f}.h5' % (model_name, pad))
    tb = TensorBoard(log_dir=log_dir)
    mcp = ModelCheckpoint(checkpoint_file, 
                          monitor='loss',
                          verbose=1,  
                          mode='auto', 
                          period=saving_period)
    
    input_size, output_size, pad_size, tile_corners = get_sizes(im_size,
                                                                n_tiles=n_tiles
                                                                )
    
    gen_d = DirectoryImgGenerator(main_dir, 
                                    im_size = im_size,
                                    weight_params=weight_params,
                                    only_contours = only_contours
                                    )
    img_generator = ImageMaskGenerator(gen_d, 
                             transform_ags, 
                             pad_size,
                             input_size,
                             tile_corners,
                             batch_size=batch_size,
                             epoch_size = batch_size*20,
                             )
    
    
    
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    
    model.fit_generator(img_generator,
                        steps_per_epoch = round(img_generator.tot_samples/batch_size), 
                        epochs = epochs,
                        verbose = 1,
                        callbacks=[tb, mcp]) #some how they were breaking the mpc ? this crashes...


import fire
if __name__ == '__main__':
    fire.Fire(main)