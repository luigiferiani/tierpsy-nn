#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:16:02 2017

@author: ajaver
"""
from keras.models import load_model

import tables
import glob
import os
import numpy as np
import time

from skimage.transform import rescale
from tierpsy.helper.misc import get_base_name, RESERVED_EXT
from augmentation import process_data, get_sizes

def get_train_test_files(prev_results=[]):
    #%%
    food_root = '/Users/ajaver/OneDrive - Imperial College London/food/'
    valid_files = glob.glob(os.path.join(food_root, 'segmentation', '**', '*_res.png'), recursive=True)
    base_names_f = [os.path.basename(x).replace('_res.png', '') for x in valid_files]
    
    all_mask_dir = [
            '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/',
            '/Users/ajaver/OneDrive - Imperial College London/optogenetics/',
            '/Volumes/behavgenom_archive$/Avelino/screening/Development/MaskedVideos/',
            '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/'
            ]

    food_train_dir = os.path.join(food_root, 'train_set_corr')
    if not os.path.exists(food_train_dir):
        os.makedirs(food_train_dir)
    
    mask_files = []
    for mask_dir in all_mask_dir:
        fnames = glob.glob(os.path.join(mask_dir, '**', '*.hdf5'), recursive=True)
        fnames = [x for x in fnames if not any(x.endswith(ext) for ext in RESERVED_EXT)]
        mask_files += fnames
    
    bnames = [(x, get_base_name(x)) for x in mask_files]
    bnames = [x for x in bnames if x[1] not in prev_results]
    
    train_files = [x for x, bn in bnames if bn in base_names_f]
    test_files = [x for x, bn in bnames if not bn in base_names_f]
    
    return train_files, test_files



def background_prediction(Xi, 
                          model_t, 
                          n_flips = 1,
                          n_tiles=4, 
                          im_size=None,
                          _is_debug=False):
    
    
    def _flip_d(img_o, nn):
        if nn == 0:
            img = img_o[::-1, :]
        elif nn == 2:
            img = img_o[:, ::-1]
        elif nn == 3:
            img = img_o[::-1, ::-1]
        else:
            img = img_o
        
        return img
    
    if im_size is None:
        im_size = Xi.shape
    
    Y_pred = np.zeros(im_size)
    for n_t in range(n_flips):
        
        X = _flip_d(Xi, n_t)
        
        if im_size is None:
            im_size = X.shape 
        input_size, output_size, pad_size, tile_corners = get_sizes(im_size)
        x_crop = process_data(X, input_size, pad_size, tile_corners) 
        x_crop = np.concatenate(x_crop)
        y_pred = model_t.predict(x_crop)
        
        
        Y_pred_s = np.zeros(X.shape)
        N_s = np.zeros(X.shape)
        for (i,j), yy,xx in zip(tile_corners, y_pred, x_crop):
            Y_pred_s[i:i+output_size, j:j+output_size] += yy[:,:,1]
            
            if _is_debug:
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(np.squeeze(xx))
                plt.subplot(1,2,2)
                plt.imshow(yy[:,:,1])
            
            N_s[i:i+output_size, j:j+output_size] += 1
        Y_pred += _flip_d(Y_pred_s/N_s, n_t)
    
    return Y_pred



if __name__ == '__main__':
    SAVE_RESULTS = True
    IS_PLOT = True
    EXPECTED_SIZE = 512
    prev_results = []
    if SAVE_RESULTS:
        save_dir  ='./bgnd_results_rig'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        prev_results = [os.path.basename(x).replace('_food.npy','') for x in os.listdir(save_dir)]
    
    train_files, test_files = get_train_test_files(prev_results)
    
    model_patch = load_model('unet_RMSprop-5-04999-0.3997.h5')
    model_border = load_model('unet_RMSprop-5_cnt-02499-1.2666.h5')
    #%%
    import random
    sample_files = test_files
    #sample_files = [x for x in train_files if 'oig' in x]
    sample_files = random.sample(test_files, 5)
    
    
    MAX_BGND = 2
    for imask, mask_file in enumerate(sample_files):
        print(imask+1, len(sample_files))
        tic = time.time()
        
        try:
            with tables.File(mask_file, 'r') as fid:
                if not '/full_data' in fid:
                    print('Missing full_data')
                    continue
                
                bgnd_o = fid.get_node('/full_data')[:MAX_BGND]
        except tables.exceptions.HDF5ExtError:
            print('Bad File')
            continue
        
        assert bgnd_o.ndim == 3
        if bgnd_o.shape[0] > 1:
            bgnd = [np.max(bgnd_o[i:i+1], axis=0) for i in range(bgnd_o.shape[0]-1)] 
        else:
            bgnd = [np.squeeze(bgnd_o)]
        
        min_size = max(bgnd[0].shape)
        resize_factor = min(EXPECTED_SIZE, min_size)/min_size
        bgnd_s = [rescale(x, resize_factor, mode='constant')*255 for x in bgnd]
        
        
        input_size, output_size, pad_size, tile_corners = get_sizes(bgnd_s[0].shape)
        
        
        for b_img in bgnd_s:
            Y_pred = []
            for mod in [model_patch, model_border]:
                yy = background_prediction(b_img, 
                                          mod, 
                                          n_flips=1,
                                          n_tiles=4
                                          )
                Y_pred.append(yy)
            
            
            if IS_PLOT:
                import matplotlib.pylab as plt
                n_rows= len(Y_pred) + 1
                plt.figure()
                plt.subplot(1,n_rows,1)
                plt.imshow(b_img, cmap='gray')
                for irow, yy in enumerate(Y_pred):
                    plt.subplot(1, n_rows, irow+2)    
                    plt.imshow(yy, interpolation='none')
        
        print(time.time() - tic)
        
        
        if SAVE_RESULTS:
            result = np.stack([b_img] + Y_pred)
            bn = get_base_name(mask_file)
            sname = os.path.join(save_dir, bn + '_food.npy')
            np.save(sname, result)
        
        
        