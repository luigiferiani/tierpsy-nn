#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:45:14 2017

@author: ajaver
"""
import os
import matplotlib.pylab as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

import sys
sys.path.append("../find_food")


from skimage.io import imread
from skimage.transform import resize
import random
import numpy as np

import keras.backend as K
from keras.utils import to_categorical

from augmentation import ImageMaskGenerator, process_data


def get_sizes(im_size, d4a_size= 24, n_conv_layers=4, n_random_tiles=None):
    ''' Useful to determine the expected inputs and output sizes of a u-net.
    Additionally if the image is larger than the network output the points to 
    subdivide the image in tiles are given
    '''
    
    #assuming 4 layers of convolutions
    def _in_size(d4a_size): 
        mm = d4a_size
        for n in range(n_conv_layers):
            mm = mm*2 + 2 + 2
        return mm

    def _out_size(d4a_size):
        mm = d4a_size -2 -2
        for n in range(n_conv_layers):
            mm = mm*2 - 2 - 2
        return mm

    
    #this is the size of the central reduced layer. I choose this value manually
    input_size = _in_size(d4a_size) #required 444 of input
    output_size = _out_size(d4a_size) #set 260 of outpu
    pad_size = int((input_size-output_size)/2)

    if any(x < output_size for x in im_size):
        msg = 'All the sides of the image ({}) must be larger or equal to ' \
                'the network output {}.'
        raise ValueError(msg.format(im_size, output_size))
    
    if n_random_tiles is None:
        #get an uniformly sampled set of tiles
        n_tiles_x = int(np.ceil(im_size[0]/output_size))
        n_tiles_y = int(np.ceil(im_size[1]/output_size))
        
        
        txs = np.round(np.linspace(0, im_size[0] - output_size, n_tiles_x)).astype(np.int)
        tys = np.round(np.linspace(0, im_size[1] - output_size, n_tiles_y)).astype(np.int)
        
        
        tile_corners = [(tx, ty) for tx in txs for ty in tys]
    else:
        x_coord = np.random.randint(0, int(im_size[0]-output_size), n_random_tiles)
        y_coord = np.random.randint(0, int(im_size[1]-output_size), n_random_tiles)
        tile_corners = [dd for dd in zip(x_coord, y_coord)]
        
    
    return input_size, output_size, pad_size, tile_corners


class DirectoryImgGenerator(object):
    def __init__(self, 
                 main_dir, 
                 im_size=None, 
                 weight_params={}
                 ):
        
        self.main_dir = main_dir
        self.weight_params = weight_params
        self.im_size = im_size
        self.weight_params = weight_params 
        fnames = os.listdir(main_dir)
        fnames = [x for x in fnames if x.endswith('.bmp')]
        self.fnames = fnames
        
        
        
    def __len__(self): 
        return len(self.fnames)

    def __getitem__(self, i):
        return self._get(self.fnames[i])

    def __iter__(self):
        for fname in self.fnames:
            yield self._get(fname)
    
    def get_random(self):
        fname = random.choice(self.fnames)
        
        return self._get(fname)
    
    def _get(self, fname):
        print(fname)
        
        im_file = os.path.join(self.main_dir, fname) 
        csv_file = os.path.splitext(im_file)[0] + '_eggs.csv'
        
        X = imread(im_file)
        Yo = np.zeros_like(X)
        
        df = pd.read_csv(csv_file)
        inds = np.ravel_multi_index([df['y'].astype(np.int), df['x'].astype(np.int)], X.shape)
        Yo.flat[inds] = 1
        
        if self.im_size is not None and X.shape != self.im_size:
            #resize refit image to be between 0-1
            X = resize(X, self.im_size, mode='reflect')*255
            Yo = resize(Yo, self.im_size, mode='reflect')>0
        
        
        
        Y = to_categorical(Yo, 2)
        Y = np.reshape(Y, (Yo.shape[0], Yo.shape[1], 2))
        
        
        
        def _normalize_weigths_by_class(_Y):
            #normalize the weights for the classes
            W_label = np.zeros(_Y.shape, K.floatx()) 
            lab_w = np.mean(_Y)
            
            dd = _Y>0
            W_label[dd] = 1/lab_w 
            W_label[~dd] = 1/(1-lab_w)
            return W_label
        
        
        
        W = _normalize_weigths_by_class(Yo)
        
        W_border = self.weight_params['weigth']*gaussian_filter(Yo.astype(K.floatx()), sigma=self.weight_params['sigma'])
        Y[:,:,0] *= W_border
        
        # I can add the weights directly to the predictions because 
        #keras uses y_true * log(y_pred) so y_true can be any number larger than 0
        #and
        # categorical_accuracy uses the maximum argument as the real prediction
        Y = Y*W[..., None]
            
            
        return X,Y



if __name__ == '__main__':
    #%%
    transform_ags = dict(
            rotation_range=90, 
             shift_range = 0.1,
             zoom_range = (0.9, 1.5),
             horizontal_flip=True,
             vertical_flip=True,
             elastic_alpha_range=None,
             elastic_sigma=None,
             int_alpha=(0.5,2.25)
             )
    
    weight_params = dict(
            sigma = 5,
            weigth = 20
            )
    
    im_size = ((2048, 2048))
    n_tiles = 16
    
    main_dir = '/Users/ajaver/OneDrive - Imperial College London/egg_counter/example/'
    
    gen_d = DirectoryImgGenerator(main_dir, 
                                  weight_params=weight_params)
    
    input_size, output_size, pad_size, tile_corners = get_sizes(im_size, n_random_tiles=n_tiles, d4a_size=24)
    
    
    gen = ImageMaskGenerator(gen_d, 
                             transform_ags, 
                             pad_size,
                             input_size,
                             tile_corners,
                             batch_size=16
                             )
#%%
    X,Y = gen_d.get_random()
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(X, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(Y[:,:,1])
    plt.subplot(1,3,3)
    plt.imshow(Y[:,:,0])
    #%%
    from functools import partial
    f_args = dict(input_size=input_size, 
                  pad_size=pad_size, 
                  tile_corners=tile_corners, 
                  transform_ags=transform_ags
                  )
        
    current_img = X,Y
    n_sets = 1
    _process_data = partial(process_data, **f_args)
    #%%
    D  = zip(*list(map(_process_data, [current_img]*n_sets))) #process data
    D = [np.concatenate(sum(x, [])) for x in D] #pack data
    D = [None if x is None else x[:16] for x in D]
    #%%
    def _add_patch(x,patch):
        I_y = np.squeeze(x.copy())
        top = np.max(I_y)
        bot = np.min(I_y)
        I_y = (I_y-bot)/(top-bot)
        
        I_y = np.repeat(I_y[:,:,None], 3, axis=2)
        
        
        top = np.max(patch)
        bot = np.min(patch)
        if top-bot > 0:
            patch = (patch-bot)/(top-bot)
            I_y[pad_size:-pad_size, pad_size:-pad_size, 1] *= 1-patch*0.5       
            I_y[pad_size:-pad_size, pad_size:-pad_size, 0] *= 0.5
        return I_y
    
    
    for nn, (batch_x, batch_y) in enumerate(gen):
        if nn > 0:
            break
        for ii, (xx,yy) in enumerate(zip(batch_x, batch_y)):
            
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(np.squeeze(xx))
            
            plt.subplot(1,3,2)
            I_y = _add_patch(xx,yy[..., 0])
            plt.imshow(I_y)
            
            plt.subplot(1,3,3)
            I_y = _add_patch(xx,yy[..., 1])
            plt.imshow(I_y)
    

