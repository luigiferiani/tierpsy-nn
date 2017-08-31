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
from find_food import get_food_contour
from tierpsy.helper.misc import get_base_name, RESERVED_EXT


if __name__ == '__main__':
    food_root = '/Users/ajaver/OneDrive - Imperial College London/food/CeNDR'
    mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/'
    
    #food_root = '/Users/ajaver/OneDrive - Imperial College London/food/optogenetics'
    #mask_dir = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/'
    
    #food_root = '/Users/ajaver/OneDrive - Imperial College London/food/Development'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Development/MaskedVideos/'
    
    #food_root = '/Users/ajaver/OneDrive - Imperial College London/food/Worm_Rig_Tests'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/'
    
    
    food_results = os.path.join(food_root, 'segmentation')
    
    n_bins = 180
    frac_lowess=0.1    
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Development/MaskedVideos/Development_C1_170617/'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Development/MaskedVideos/**/'
    
    
    fnames = glob.glob(os.path.join(mask_dir, '**', '*.hdf5'), recursive=True)
    fnames = [x for x in fnames if not any(x.endswith(ext) for ext in RESERVED_EXT)]
    
    if not os.path.exists(food_results):
        os.makedirs(food_results)
    
    for mask_video in fnames:
        circx, circy = get_food_contour(mask_video, 
                                        n_bins = n_bins,
                                        frac_lowess=frac_lowess,
                                        is_debug=True)
        food_cnt = np.vstack((circx, circy)).T
        #%%
        base_name = get_base_name(mask_video)
        with tables.File(mask_video, 'r') as fid:
            full_data = fid.get_node('/full_data')[0]
        
        #%%
        import matplotlib.pylab as plt
        fig = plt.figure(figsize=(10,10))
        plt.imshow(full_data, cmap='gray')
        plt.plot(circy, circx)
        
        res_plot_f = os.path.join(food_results, base_name + '_res.png')
        fig.savefig(res_plot_f)
        
        plt.close(fig)
        #%%
        f_cnt = os.path.join(food_results, base_name + '_cnt.npy')
        np.save(f_cnt, food_cnt)
        
        
#%%
#base_name = get_base_name(mask_video)
#        with tables.File(mask_video, 'r') as fid:
#            full_data = fid.get_node('/full_data')[0]
#    full_max = np.max(full_data, axis=0)
#    full_min = np.min(full_data, axis=0)
        
        
        