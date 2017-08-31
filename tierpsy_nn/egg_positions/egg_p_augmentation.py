#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:45:14 2017

@author: ajaver
"""
import os
import matplotlib.pylab as plt
import pandas as pd

from functools import partial
from skimage.io import imread
from skimage.transform import resize
import random
import numpy as np
import cv2

from keras.preprocessing.image import Iterator

import sys
dname = os.path.dirname(__file__)
sys.path.append(os.path.join(dname, "..", "find_food"))

from augmentation import process_data


def get_sizes(im_size, d4a_size= 24, n_conv_layers=4):
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
    
    n_tiles_x = int(np.ceil(im_size[0]/output_size))
    n_tiles_y = int(np.ceil(im_size[1]/output_size))
    
    
    txs = np.round(np.linspace(0, im_size[0] - output_size, n_tiles_x)).astype(np.int)
    tys = np.round(np.linspace(0, im_size[1] - output_size, n_tiles_y)).astype(np.int)
    
    
    tile_corners = [(tx, ty) for tx in txs for ty in tys]
    return input_size, output_size, pad_size, tile_corners


class DirectoryImgGenerator(object):
    def __init__(self, 
                 main_dir
                 ):
        
        fnames = os.listdir(main_dir)
        fnames = [x for x in fnames if x.endswith('.bmp')]
        if not fnames:
            raise ValueError("Directory {} does not contain .bmp files.".format(main_dir))
        
        
        self.main_dir = main_dir
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
        df = pd.read_csv(csv_file)
        
        coords = [df['y'].values.astype(np.int), df['x'].values.astype(np.int)]
        
        return X, coords

class ImageMaskGenerator(Iterator):
    
    def __init__(self, 
                 generator,
                 transform_ags,
                 pad_size,
                 input_size,
                 n_tiles,
                 batch_size=32, 
                 epoch_size=None,
                 shuffle=True, 
                 seed=None):
       
        self.generator = generator
        
        if epoch_size is None:
            self.tot_samples = len(self.generator)
        else:
            self.tot_samples = epoch_size
        
        self.transform_ags = transform_ags
        self.pad_size = pad_size
        self.input_size = input_size
        self.output_size = input_size - pad_size*2
        self.batch_size = batch_size
        self.n_tiles = n_tiles
        #i really do not use this functionality i could reimplement it in the future
        super(ImageMaskGenerator, self).__init__(self.tot_samples, batch_size, shuffle, seed)

    def next(self):
        """
        # Returns
            The next batch.
        """
        
        #with self.lock:
        #    index_array, current_index, current_batch_size = next(self.index_generator)
        #I will read a only one image and do transforms until i get the n_batches
        #current_img = self.generator[index_array[0]]
        #batch_size = index_array.size
        Xo, coords = self.generator.get_random()
        im_size = Xo.shape
        
        X = Xo/255
        X -= np.median(X)
        
        inds = np.ravel_multi_index(coords, im_size)
        
        lab_w = inds.size/Xo.size
        
        
        Yo_b = np.full(Xo.shape, 1/(1-lab_w), np.float32)
        Yo_b.flat[inds] = 0
        
        Yo_a = np.zeros(Xo.shape, np.float32)
        Yo_a.flat[inds] = 1/lab_w
        Y = np.stack((Yo_a, Yo_b))
        
           #normalize the weights for the classes
        
        W = 20/(1+cv2.distanceTransform((Yo_b>0).astype(np.uint8),cv2.DIST_C,0)/2)
        Yo_b *= W
        
        Y = np.stack((Yo_b, Yo_a), axis=2)
        
        
        centers = np.array(random.sample(list(zip(*coords)), self.n_tiles))
        tile_corners = centers -  np.random.randint(0, self.output_size, centers.shape)
        tile_corners[:,0] = np.clip(tile_corners[:,0], 0,  int(im_size[0]-self.output_size))
        tile_corners[:,1] = np.clip(tile_corners[:,1], 0,  int(im_size[1]-self.output_size))
        
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        f_args = dict(input_size=self.input_size, 
                          pad_size=self.pad_size, 
                          tile_corners=tile_corners, 
                          transform_ags=self.transform_ags
                          )
        
        n_sets = int(np.ceil(self.batch_size/self.n_tiles))
        _process_data = partial(process_data, **f_args)
        D  = zip(*list(map(_process_data, [(X,Y)]*n_sets))) #process data
        D = [np.concatenate(sum(x, [])) for x in D] #pack data
        D = [None if x is None else x[:self.batch_size] for x in D]
        return D

if __name__ == '__main__':
    #%%
#    transform_ags = dict(
#            rotation_range=90, 
#             shift_range = 0.1,
#             zoom_range = (0.9, 1.5),
#             horizontal_flip=True,
#             vertical_flip=True,
#             elastic_alpha_range=None,
#             elastic_sigma=None,
#             int_alpha=(0.5,2.25)
#             )
    
    transform_ags = {}
    
    im_size = ((2048, 2048))
    n_tiles = 32
    
    main_dir = '/Users/ajaver/OneDrive - Imperial College London/egg_counter/example/'
    
    gen_d = DirectoryImgGenerator(main_dir)
    
    input_size, output_size, pad_size, tile_corners = get_sizes(im_size, d4a_size=24,  n_conv_layers=3)
    
    
    gen = ImageMaskGenerator(gen_d, 
                             transform_ags, 
                             pad_size,
                             input_size,
                             n_tiles = 16,
                             batch_size=16
                             )

    X,coords = gen_d.get_random()
#    
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.imshow(X, cmap='gray')
#    plt.subplot(1,2,2)
#    plt.imshow(Y)

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
            I_y[pad_size:-pad_size, pad_size:-pad_size, 1] *= 1-patch       
            I_y[pad_size:-pad_size, pad_size:-pad_size, 0] *= 0.5
        return I_y
    
    
    for nn, (batch_x, batch_y) in enumerate(gen):
        if nn > 0:
            break
        for ii, (xx,yy) in enumerate(zip(batch_x, batch_y)):
            
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.imshow(np.squeeze(xx))
            
            plt.subplot(1,2,2)
            I_y = _add_patch(xx, np.squeeze(yy[..., 0]))
            plt.imshow(I_y)
            

