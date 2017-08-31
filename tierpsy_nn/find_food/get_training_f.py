#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:30:50 2017

@author: ajaver
"""
import tables
import glob
import os
import numpy as np
import cv2
from tierpsy.helper.misc import get_base_name, RESERVED_EXT


def _resize_img(img, dsize, expected_size):
    img = cv2.resize(img, dsize)
    
    if img.shape[0] != expected_size:
        ini = (img.shape[0]-expected_size)//2
        img = img[ini:(ini+expected_size), :]
        
        shift_d = (ini, 0)
        
    elif img.shape[1] != expected_size:
        ini = (img.shape[1]-expected_size)//2
        img = img[:, ini:(ini+expected_size)]
        shift_d = (0, ini)
    else:
        shift_d = (0,0)
    
    return img, shift_d
    
if __name__ == '__main__':
    is_debug = False
    expected_size = 512
    food_root = '/Users/ajaver/OneDrive - Imperial College London/food/'
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
    
    valid_files = glob.glob(os.path.join(food_root, 'segmentation', '**', '*_res.png'), recursive=True)
    cnt_files = [x.replace('_res.png', '_cnt.npy') for x in valid_files]
    base_names_f = [os.path.basename(x).replace('_res.png', '') for x in valid_files]
    cnt_dict = {bn:x for bn,x in zip(base_names_f, cnt_files)}
    base_names_m = [os.path.basename(x).replace('.hdf5', '') for x in mask_files]
    mask_dict = {bn:x for bn,x in zip(base_names_m, mask_files)}
    
    
    fnames = [(cnt_dict[bn], mask_dict[bn]) for bn in cnt_dict]# if bn in mask_dict]
    for ii, (cnt_file, mask_file) in enumerate(fnames):
        base_name = get_base_name(mask_file)
        
        if not os.path.exists(cnt_file) or not os.path.exists(mask_file):
            continue
        
        with tables.File(mask_file, 'r') as fid:
            full_data = fid.get_node('/full_data')[:]
            
            min_size = min(full_data.shape[1:])
            resize_factor = min(expected_size, min_size)/min_size
            dsize = tuple(int(x*resize_factor) for x in full_data[0].shape[::-1])
            
            D = [_resize_img(img, dsize, expected_size) for img in full_data]
            full_data, shift_d = map(list, zip(*D))
            
            
            
        
        assert all(shift_d[0]==x for x in shift_d)
        shift_d = np.array(shift_d[0])
            
        full_max = np.max(full_data, axis=0)
        full_min = np.min(full_data, axis=0)
        
        if np.median(full_min) < 1:
            print('bad')
            continue
        
        #get ground truth
        food_cnt = np.load(cnt_file)
        food_cnt *=  resize_factor
        food_cnt -= shift_d
        food_cnt = food_cnt[:, ::-1]
        
        mask = np.zeros(full_max.shape, np.uint8)
        pts = np.round(food_cnt).astype(np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(mask, [pts], 255)
        
        assert all(x == expected_size for x in mask.shape)
        
        for nn, img in enumerate([full_max, full_min] + full_data):
            ff = '_{}_{}.png'.format(nn, base_name)
            f2save_x = os.path.join(food_train_dir, 'X'+ff)
            f2save_y = os.path.join(food_train_dir, 'Y'+ff)
            
            cv2.imwrite(f2save_x, img)
            cv2.imwrite(f2save_y, mask)
        
        if is_debug:
            import matplotlib.pylab as plt
            dd = full_max.copy()
            dd[mask==1] = 255
            plt.figure()
            plt.imshow(dd, cmap='gray')
        
        print('{} of {} {}'.format(ii+1, len(fnames), base_name))
        
        
