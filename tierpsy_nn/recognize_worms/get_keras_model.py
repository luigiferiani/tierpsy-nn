#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:24:40 2016

@author: ajaver
"""

import tables
import numpy as np
import os
import time

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam



from tierpsy.analysis.ske_init.filterTrajectModel import shift_and_normalize

def read_field_data(filename, field):
    with tables.File(filename, 'r') as fid:
        data = fid.get_node('/', field + '_x')[:]
        label = fid.get_node('/', field + '_y')[:]
    
        
    #if rescale != 1:
    #    data = np.array([imresize(x, rescale) for x in data])
    
    dat_x = shift_and_normalize(data)[: ,:, :, np.newaxis]
    
    
    #a worm will be 2 (good worm),3(difficult worm),4 (aggregate)
    good_labels = ((label != 1) & (label <= 4)).astype(np.uint8)
    dat_y = np_utils.to_categorical(good_labels, 2)
    
    return dat_x, dat_y


#%%
def get_model_old(X_train, Y_train, X_val, Y_val):
    
    rand_seed = 1337
    np.random.seed(rand_seed)  # for reproducibility
    
    
    batch_size = 128
    nb_classes = 2
    nb_epoch = 50
    
    # input image dimensions
    img_rows, img_cols = X_train.shape[1:3]
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (5, 5)
    
    
    model = Sequential()
    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape= (img_rows, img_cols, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(70))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_val, Y_val))
    return model

def get_model(roi_size, nb_classes):
    name = ''
    
    input_shape = (roi_size, roi_size, 1)
    input_data = Input(shape=input_shape, name='input_all')
    
    
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
    
    x = GlobalMaxPooling2D(name=name+'avg_pool')(x)
    
    x = Dense(512, name=name+'dense0', activation='relu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(64, name=name+'dense1', activation='relu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)
    
    
    model = Model(input_data, x)
    optimizer = Adam(lr=1e-4)#, decay=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    filename = '/Users/ajaver/OneDrive - Imperial College London/training_data/sample.hdf5'
    model_trained_path = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker/misc/model_isworm_20161130_002654.h5'
    
    data = {lab:read_field_data(filename, lab) for lab in ['train', 'val']}
    
    roi_size = data['train'][0].shape[1]
    nb_classes = data['train'][1].shape[1]
    model = get_model(roi_size, nb_classes)
    #%%
    SAVE_DIR = '/Volumes/behavgenom_archive$/Avelino/is_worm/'
    
    batch_size = 128
    epochs = 200
    
    log_dir = os.path.join(SAVE_DIR, 'logs', 'main_%s' % time.strftime('%Y%m%d_%H%M%S'))
    pad=int(np.ceil(np.log10(epochs+1)))
    checkpoint_file = os.path.join(log_dir, 'main_resized-{epoch:0%id}-{val_acc:.4f}.h5' % pad)


    
    tb = TensorBoard(log_dir=log_dir)
    mcp = ModelCheckpoint(checkpoint_file, verbose=0,  
                          mode='auto', 
                          monitor='val_acc',
                          save_best_only=True,
                          period=1)
    
    datagen = ImageDataGenerator(rotation_range=90.,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True)
    
    model.fit_generator(datagen.flow(*data['train'], batch_size=32),
                    steps_per_epoch=len(data['train'][0])/32, 
                    epochs=epochs,
                    verbose=1,
                    validation_data=data['val'],
                    callbacks=[tb, mcp])
    
#    model = get_model_convnet(X_train, Y_train, X_val, Y_val)
#    
#    #%%
#    score = model.evaluate(X_val, Y_val, verbose=0)
#    print('Test score:', score[0])
#    print('Test accuracy:', score[1])
#    
    
#    #%%
    model_name = 'model_isworm_%s.h5' % time.strftime('%Y%m%d_%H%M%S')    
    model.save(model_name)
    #%% Show wrongly classified
#    model = load_model(model_trained_path)
#    show_bad(model, X_val, Y_val)
    #show_bad(model, X_train, Y_train)
    #X_test, Y_test = read_field_data(filename, 'test')
    

#%%
#show_images(X, labels = Y)


