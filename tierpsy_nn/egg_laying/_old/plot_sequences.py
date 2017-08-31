#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:47:24 2017

@author: ajaver
"""

import h5py
import matplotlib.pylab as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from egg_train_model import read_field_data

n_rows = 2

#sample_file = 'samples_eggs_fixed.hdf5'
sample_file = 'samples_eggs_resized.hdf5'
data = read_field_data(sample_file, 'train', tot=n_rows)

#for irow in range(n_rows):
#    seq_worm = data[0][irow]
#    seq_size = seq_worm.shape[-1]
#    for ii in range(seq_size):
#        nn = ii+1 + seq_size*irow
#        plt.subplot(n_rows, seq_size, nn)
#        plt.imshow(seq_worm[:,:,ii], interpolation='none', cmap='gray')
#        plt.axis('off')

        
#%%
datagen = ImageDataGenerator(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True)

#datagen = ImageDataGenerator(
#                     horizontal_flip=True,
#                     vertical_flip=True)
#dd = datagen.flow(data[0][:,:,:,1:], data[1], batch_size=32)
dd = datagen.flow(data[0][:,:,:,:], data[1], batch_size=32)

for idat, (X, Y) in enumerate(dd):
    for iseq in range(X.shape[0]):
        seq_worm = X[iseq]
        if iseq == 0:
            plt.figure()
        irow = iseq % 10
        
        seq_size = seq_worm.shape[-1]
        for ii in range(seq_size):
            nn = ii+1 + seq_size*irow
            plt.subplot(n_rows, seq_size, nn)
            plt.imshow(seq_worm[:,:,ii], interpolation='none', cmap='gray')
            plt.axis('off')
            
    if idat >= 5:
        break