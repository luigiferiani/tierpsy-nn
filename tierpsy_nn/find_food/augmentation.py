#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:39:27 2017

@author: ajaver
"""
import numpy as np
import random
from scipy.ndimage.interpolation import map_coordinates, affine_transform
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import dilation, disk, skeletonize, binary_erosion
from skimage.io import imread
from skimage.transform import resize
from skimage.draw import polygon

import os
import multiprocessing as mp
from functools import partial

import keras.backend as K
from keras.preprocessing.image import Iterator
from keras.utils import to_categorical


#%%
def random_rotation(rg, h, w):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)    
    return transform_matrix


def random_shift(shift_range, h, w):
    tx = np.random.uniform(-shift_range, shift_range) * h
    ty = np.random.uniform(-shift_range, shift_range) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    return translation_matrix


def random_zoom(zoom_range, h, w, same_zoom=False):
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        
    if same_zoom:
        zx = zy
    
    zoom_matrix = np.array([[1/zx, 0, 0],
                            [0, 1/zy, 0],
                            [0, 0, 1]])

    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    
    return transform_matrix

def apply_transform_img(x,
                    transform_matrix,
                    fill_mode='reflect',
                    cval=0.):
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    xt = affine_transform(x,
                        final_affine_matrix,
                        final_offset,
                        order=0,
                        mode=fill_mode,
                        cval=cval
                        )
    return xt


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def elastic_transform(h, w, alpha_range, sigma):
    alpha = np.random.uniform(0, alpha_range)
    random_state = np.random.RandomState(None)
    
    dx = gaussian_filter((random_state.rand(h, w ) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(h, w ) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x,y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x = x + dx 
    y = y + dy 
    
    elastic_inds = np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))
    
    return elastic_inds
    
    
def random_transform(h, 
                     w, 
                     rotation_range, 
                     shift_range,
                     zoom_range,
                     horizontal_flip,
                     vertical_flip,
                     elastic_alpha_range,
                     elastic_sigma,
                     same_zoom = False
                     ):
    
    rot_mat = random_rotation(rotation_range, h, w)
    shift_mat = random_shift(shift_range, h, w)
    zoom_mat = random_zoom(zoom_range, h, w, same_zoom)
    
    
    transform_mat = np.dot(shift_mat, rot_mat)
    transform_mat = np.dot(transform_mat, zoom_mat)
    
    if elastic_alpha_range is not None and elastic_sigma is not None:
        elastic_inds = elastic_transform(h, w, elastic_alpha_range, elastic_sigma)
    else:
        elastic_inds = None
        
    is_h_flip =  horizontal_flip and np.random.random() < 0.5
    is_v_flip =  vertical_flip and np.random.random() < 0.5
        

    return transform_mat, is_h_flip, is_v_flip, elastic_inds

def _transform_img(img, transform_matrix, is_h_flip, is_v_flip, elastic_inds):
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    img_aug = affine_transform(
            img,
            final_affine_matrix,
            final_offset,
            order=0,
            mode='reflect',
            output=np.float32,
            cval=0.)
    if is_h_flip:
        img_aug = img_aug[::-1, :] 
    if is_v_flip:
        img_aug = img_aug[:, ::-1] 
    
    if elastic_inds is not None:
        img_aug = map_coordinates(img_aug, elastic_inds, order=1).reshape((img.shape))
    
    return img_aug

def transform_img(D,
                  transform_matrix, 
                  is_h_flip, 
                  is_v_flip, 
                  elastic_inds
                  ):
    if D is None:
        return None
    D_aug = np.zeros_like(D)
    for nn in range(D.shape[-1]):
        D_aug[:, :, nn] = _transform_img(D[: ,:, nn], 
                      transform_matrix, 
                      is_h_flip, 
                      is_v_flip, 
                      elastic_inds)
    D_aug = np.array(D_aug)
    return D_aug

#%%
def process_data(images, 
                 input_size, 
                 pad_size, 
                 tile_corners, 
                 int_alpha = None,
                 transform_ags={},
                 ):
    
    def _get_tile_in(img, x,y):
            return img[np.newaxis, x:x+input_size, y:y+input_size, :]
        
    def _get_tile_out(img, x,y):
        #not very efficient, but i cannot be bother to fix it
        D  = _get_tile_in(img,x,y)
        return  D[:, pad_size:-pad_size, pad_size:-pad_size, :]
       
    def _cast_tf(D):
        D = D.astype(K.floatx())
        if D.ndim == 2:
            D = D[..., None]
        return D
    
    #read inputs
    Y = None
    if len(images) == 2:
        X, Y = images
    else:
        X = images
    
    
    #normalize image
    X = _cast_tf(X)
    X /= 255
    X -= np.median(X)
    
    if Y is not None:
        Y = _cast_tf(Y)
     
    pad_size_s =  ((pad_size,pad_size), (pad_size,pad_size), (0,0))
    X,Y = [None if D is None else np.lib.pad(D, pad_size_s, 'reflect') for D in [X,Y]]
    
    if 'int_alpha' in transform_ags:
        transform_ags = transform_ags.copy()
        int_alpha = transform_ags['int_alpha']
        del transform_ags['int_alpha']
    
    if len(transform_ags) > 0:
        expected_size = X.shape[:2] #the expected output size after padding
        transforms = random_transform(*expected_size, **transform_ags)
        X,Y = [transform_img(x, *transforms) for x in [X,Y]]
        
        if int_alpha is not None:
            alpha = np.random.uniform(int_alpha[0], int_alpha[1])
            X *= alpha
            
    X = [_get_tile_in(X, x, y) for x,y in tile_corners]
    
    if Y is not None:
        Y = [_get_tile_out(Y, x, y) for x,y in tile_corners]
    
    output = [x for x in (X,Y) if not x is None]
    if len(output) == 1:
        output = output[0]
    return output

class ImageMaskGenerator(Iterator):
    
    def __init__(self, 
                 generator,
                 transform_ags,
                 pad_size,
                 input_size,
                 tile_corners,
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
        self.tile_corners = tile_corners
        self.batch_size = batch_size
        
        #i really do not use this functionality i could reimplement it in the future
        super(ImageMaskGenerator, self).__init__(self.tot_samples, batch_size, shuffle, seed)

    def next(self):
        """
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        f_args = dict(input_size=self.input_size, 
                          pad_size=self.pad_size, 
                          tile_corners=self.tile_corners, 
                          transform_ags=self.transform_ags
                          )
        #with self.lock:
        #    index_array, current_index, current_batch_size = next(self.index_generator)
        #I will read a only one image and do transforms until i get the n_batches
        #current_img = self.generator[index_array[0]]
        #batch_size = index_array.size
        current_img = self.generator.get_random()
        n_sets = int(np.ceil(self.batch_size/len(self.tile_corners)))
        _process_data = partial(process_data, **f_args)
        D  = zip(*list(map(_process_data, [current_img]*n_sets))) #process data
        D = [np.concatenate(sum(x, [])) for x in D] #pack data
        D = [None if x is None else x[:self.batch_size] for x in D]
        return D

class DirectoryImgGenerator(object):
    def __init__(self, 
                 main_dir, 
                 im_size=None, 
                 weight_params={},
                 only_contours=False
                 ):
        
        self.main_dir = main_dir
        self.weight_params = weight_params
        self.im_size = im_size
        self.only_contours=only_contours
        
        fnames = os.listdir(main_dir)
        fnames = [x for x in fnames if x.endswith('.png')]
        fnames = sorted(set(x[2:] for x in fnames if x.startswith('X_')))
        
        self.fnames = fnames
        #group files by date
        group_dates = {}
        for x  in self.fnames:
            date_str = x.split('_')[-2]
            if not date_str in group_dates:
                group_dates[date_str] = {}
            
            bn = '_'.join(x.split('_')[1:-2])
            if not bn in group_dates[date_str]:
                group_dates[date_str][bn] = []
            
            group_dates[date_str][bn].append(x)
        #I am only including dates that have at least 10 different basenames (plates)
        #group_dates = {x:dat for x, dat in group_dates.items() if len(dat)>10}
        
        #%%
        self.group_dates = group_dates

    def __len__(self): 
        return len(self.fnames)

    def __getitem__(self, i):
        return self._get(self.fnames[i])

    def __iter__(self):
        for fname in self.fnames:
            yield self._get(fname)
    
    def get_random(self):
        date_str = random.choice(list(self.group_dates.keys()))
        bn = random.choice(list(self.group_dates[date_str].keys()))
        fname = random.choice(self.group_dates[date_str][bn])
        
        return self._get(fname)
    
    def _get(self, fname):
        print(fname)
        
        x_name = os.path.join(self.main_dir, 'X_' + fname) 
        y_name = os.path.join(self.main_dir, 'Y_' + fname) 
        #%%
        X = imread(x_name)
        Yo = imread(y_name)
        
        if self.im_size is not None:
            #resize refit image to be between 0-1
            X = resize(X, self.im_size, mode='reflect')*255
            Yo = resize(Yo, self.im_size, mode='reflect')>0
        
        if self.only_contours:
            Yc = Yo ^ binary_erosion(Yo) #xor operator
            Yo = dilation(Yc, disk(1))
        
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
        
        def _increase_border_weight(_Y):
            sigma = self.weight_params['sigma']
            weigth = self.weight_params['weigth']
            Yc = _Y ^ binary_erosion(_Y) #xor operator
            #increase the weights in the border
            W_border = gaussian_filter(Yc.astype(K.floatx()), sigma=2.5)
            W_border *= (sigma**2)*weigth #normalize weights
            return W_border
        
        
        W = _normalize_weigths_by_class(Yo)
        if self.weight_params:
            W_border = _increase_border_weight(Yo)
            W += W_border
            
        # I can add the weights directly to the predictions because 
        #keras uses y_true * log(y_pred) so y_true can be any number larger than 0
        #and
        # categorical_accuracy uses the maximum argument as the real prediction
        Y = Y*W[..., None]
            
            
        return X,Y

#%%
def get_sizes(im_size, d4a_size= 24, n_tiles=4):
    #assuming 4 layers of convolutions
    def _in_size(d4a_size, N = 4): 
        mm = d4a_size
        for n in range(N):
            mm = mm*2 + 2 + 2
        return mm

    def _out_size(d4a_size, N = 4):
        mm = d4a_size -2 -2
        for n in range(N):
            mm = mm*2 - 2 - 2
        return mm


    #this is the size of the central reduced layer. I choose this value manually
    input_size = _in_size(d4a_size, N = 4) #required 444 of input
    output_size = _out_size(d4a_size, N = 4) #set 260 of outpu
    pad_size = int((input_size-output_size)/2)

    if n_tiles == 1:
        tile_corners = [(0,0)]
    else:
        ty = im_size[1]-output_size
        tx = im_size[0]-output_size
        
        
        if n_tiles < 4:
            tile_corners = []
        else:
            tile_corners = [(0,0), 
                            (0, ty),
                            (tx, 0),
                            (tx,ty)
                            ] #corners on how the image is going to be subdivided
        
        
        
            
        if n_tiles == 5:
            tile_corners.append((pad_size,pad_size))
        
        if len(tile_corners) != n_tiles:
            nn = n_tiles-len(tile_corners)
            extra_tiles = np.random.randint(0, int(im_size[0]-output_size), (nn, 2))
            tile_corners += [tuple(x) for x in extra_tiles]

    return input_size, output_size, pad_size, tile_corners
#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt

        
    #im_size = (260,260)
    #n_tiles=1
    
    im_size = (512, 512)
    n_tiles=8
    
    only_contours = True
    main_dir = os.path.join(os.environ['HOME'], 'OneDrive - Imperial College London/training_data/food/train_set')
    
    
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
            weigth=10
            )
    
    
    
    #input_size, output_size, pad_size, tile_corners = get_sizes(im_size, n_tiles=n_tiles)
    
    
    
    gen_d = DirectoryImgGenerator(main_dir, 
                                  only_contours=only_contours,
                                  im_size = im_size,
                                  weight_params = weight_params
                                  )
    
    gen = ImageMaskGenerator(gen_d, 
                             transform_ags, 
                             pad_size,
                             input_size,
                             tile_corners,
                             batch_size=8)
    
    
    assert gen.output_size == output_size
    #%%
    for ii in range(1):
        X, Y = gen_d.get_random()
        
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(X), cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(np.squeeze(Y[...,0]))
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(Y[...,1]))


    #%%
    for nn, (batch_x, batch_y) in enumerate(gen):
        if nn > 10:
            break
    
        for ii, (X, Y) in enumerate(zip(batch_x, batch_y)):
            #%%
            xx = np.squeeze(X)
            bot = np.min(xx)
            top = np.max(xx)
            
            ybot = np.min(Y)
            ytop = np.max(Y)
            yy_n = (Y-ybot)/(ytop-ybot)
            
            
            xi = (((xx-np.min(xx))*255)).astype(np.uint)
            xi = np.clip(xi, 0, 255)
            
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(xi, cmap='gray')
            
            plt.subplot(1,3,2)
            I_y = xx.copy()
            patch = (yy_n[:,:,0]*(top-bot))+bot
            I_y[gen.pad_size:-gen.pad_size, gen.pad_size:-gen.pad_size] = patch        
            plt.imshow(I_y)
            
            plt.subplot(1,3,3)
            
            I_w = xx.copy()
            patch = (yy_n[:,:,1]*(top-bot))+bot
            I_w[gen.pad_size:-gen.pad_size, gen.pad_size:-gen.pad_size] = patch        
            plt.imshow(I_w)
            
            #%%
            break