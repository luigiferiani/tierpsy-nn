#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:13:12 2017

@author: ajaver
"""

import matplotlib.pylab as plt
import numpy as np
import tables
import time
import os

from keras import layers
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Lambda

from keras.utils import np_utils
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

#from egg_trainset import _plot_seq
from keras.models import Model

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

rand_seed = 1337
np.random.seed(rand_seed)  # for reproducibility
#%%
def _model_single_image(img_input, name):
    name += '_'
    
    x = Conv2D(32, (3, 3), padding='same', name=name+'conv0')(img_input)
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
    
    x = GlobalMaxPooling2D(name=name+'avg_pool')(x)
    
    x = Dense(512, name=name+'dense0', activation='elu')(x)
    x = Dropout(0.4)(x)
    
    return x
    
#%%
def model_window(win_size, roi_size, nb_classes):
    win_size = 4
    roi_size = 100
    nb_classes = 2
    
    input_shape = (win_size, roi_size, roi_size,1)
    input_data = Input(shape=input_shape, name='input_all')
    
    in_layers = [Lambda(lambda x: x[:,ii,:,:], output_shape=(roi_size,roi_size,1))(input_data) for ii in range(win_size)]
    out_layers = [_model_single_image(in_layer, 'input' + str(ii)) for ii, in_layer in enumerate(in_layers)]
    x = layers.add(out_layers)
    x = Dense(512, name='joined0', activation='elu')(x)
    x = Dense(128, name='joined1', activation='elu')(x)
    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)
        
    model = Model(input_data, x)
    optimizer = Adam(lr=1e-3)#, decay=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
#%%
def model_separable(win_size, roi_size, nb_classes):
    name = ''
    
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
    
    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)
        
    model = Model(input_data, x)
    optimizer = Adam(lr=1e-4)#, decay=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


#%%
def read_field_data(filename, field, tot=None):
    with tables.File(filename, 'r') as fid:
        indexes = fid.get_node('/partitions/' + field)
        if tot is None:
            tot = indexes.shape[0]
        
        indexes = indexes[:tot]
        
        X = fid.get_node('/egg_laying_X')[indexes, :, :, :]
        #X = X[:,:, :, :, np.newaxis]
        X = np.rollaxis(X, 1, 4)
        X = X[:, :, :, :-1]
        
        
        dat = fid.get_node('/egg_laying_Y')[indexes]
        Y = dat==0
        Y = np_utils.to_categorical(Y, 2)
        
    return X, Y
#%%
def show_bad(model, X, Y):
    pred = model.predict_proba(X)
    
    label_pred = np.argmax(pred, axis=1)
    label_real = np.argmax(Y, axis=1)    
    
    bad_labels = label_pred!=label_real
    bad_ind, = np.where(bad_labels)
    print('*****')
    print(len(bad_ind), len(label_real))
    return bad_ind
#%%
if __name__ == '__main__':
    SAVE_DIR = '/Volumes/behavgenom_archive$/Avelino/eggs_tests/'
    
    training_file = 'samples_eggs_resized.hdf5'
    training_file = os.path.join(SAVE_DIR, 'data', training_file)
    #filename = '/Users/ajaver/OneDrive - Imperial College London/training_data/sample.hdf5'
    #model_trained_path = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker/misc/model_isworm_20161130_002654.h5'
    
    
    tot_dat = None
    data = {}
    for field in ['test', 'train', 'val']:
        data[field] = read_field_data(training_file, field, tot=tot_dat)
    
    # input image dimensions
    #_, win_size, img_rows, img_cols, _ = X_train.shape
    _, roi_size, _, win_size = data['train'][0].shape
    nb_classes = data['train'][1].shape[1]
    
    #model = model_window(win_size, roi_size, nb_classes)
    model = model_separable(win_size, roi_size, nb_classes)
    #%%
    batch_size = 128
    epochs = 200
    
    
    
    log_dir = os.path.join(SAVE_DIR, 'logs', 'main_%s' % time.strftime('%Y%m%d_%H%M%S'))
    pad=int(np.ceil(np.log10(epochs+1)))
    checkpoint_file = os.path.join(log_dir, 'main_resized-{epoch:0%id}-{loss:.4f}.h5' % pad)

    
    tb = TensorBoard(log_dir=log_dir)
    mcp = ModelCheckpoint(checkpoint_file, verbose=0,  mode='auto', period=1)
    
    datagen = ImageDataGenerator(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True)
    
    model.fit_generator(datagen.flow(*data['train'], batch_size=32),
                    steps_per_epoch=len(data['train'][0]), 
                    epochs=epochs,
                    verbose=1,
                    validation_data=data['val'],
                    callbacks=[tb, mcp])

    #%%
    X_test, Y_test = read_field_data(training_file, 'test')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    score = model.evaluate(*data['val'], verbose=0)
    print('Validation score:', score[0])
    print('Validation accuracy:', score[1])
    
    #%%
    bad_ind = show_bad(model, *data['val'])
#    for ind in bad_ind:
#            bad_seq = np.rollaxis(X[ind], 2, -3)
#            _plot_seq(bad_seq)
    #%%
#    for ind in range(2840, 2855):
#        mod = X_train[ind]
#        prob = model.predict_proba(mod[np.newaxis, :, :, :])
#        bad_seq = np.rollaxis(mod, 2, -3)
#        plt.figure()
#        _plot_seq(bad_seq)
#        plt.suptitle('{} | {}'.format(Y_train[ind], np.round(prob)))
    #%%
    
#    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=10,
#              verbose=1, validation_data=(X_val, Y_val))
#    model_name = 'model_egg_laying_%s.h5' % time.strftime('%Y%m%d_%H%M%S')    
#    model.save(model_name)
    