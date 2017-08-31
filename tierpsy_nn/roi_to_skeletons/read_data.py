#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:03:28 2017

@author: ajaver
"""
import tables
import numpy as np
import pandas as pd
import cv2
import os

from tierpsy.analysis.feat_create.obtainFeaturesHelper import _h_smooth_curve_all
from tierpsy.analysis.ske_create.helperIterROI import generateMoviesROI
from tierpsy.analysis.ske_init.filterTrajectModel import shift_and_normalize
import multiprocessing as mp


def read_data_generator(masked_file, trajectories_data):
    roi_generator = generateMoviesROI(masked_file, trajectories_data)
    for worms_in_frame in roi_generator:
        for ind, roi_dat in worms_in_frame.items():
            row_data = trajectories_data.loc[ind]
            worm_img, roi_corner = roi_dat
            skeleton_id = int(row_data['skeleton_id'])            
            yield (ind, worm_img, roi_corner, skeleton_id)


def generate_chunck(generator_single, chuck_size):
    dat = []
    for row in generator_single:
        dat.append(row)
        if len(dat) >= chuck_size:
            yield dat
            dat = []
    yield dat

def process_chuck(chuck):
    
    inds, worm_imgs, roi_corners, skeleton_ids = zip(*chuck)
    with tables.File(skeletons_file, 'r') as fid:
        skel_o = fid.get_node('/skeleton')[skeleton_ids, : ,:]
    
    roi_corners = np.array(roi_corners)
    scale_factors = np.array([roi_size/x.shape[0] for x in worm_imgs])
    skel_o -= roi_corners[:, np.newaxis, :]
    skel_sc = _h_smooth_curve_all(skel_o)*scale_factors[:, np.newaxis, np.newaxis]
    
    
    worm_imgs_n = [shift_and_normalize(x.astype(np.float32)) for x in worm_imgs]
    worm_img_sc = [cv2.resize(x, (roi_size,roi_size)) for x in worm_imgs_n]
    
    
    return inds, skel_sc, worm_img_sc

def _init_table(save_file, trajectories_data):
    tot_rows = len(trajectories_data)
    trajectories_data.index = np.arange(tot_rows)
    
    TABLE_FILTERS = tables.Filters(
                    complevel=5, complib='zlib', shuffle=True, fletcher32=True)
    with tables.File(save_file, 'w') as fid:
        rec_data = trajectories_data.to_records(index=False)
        rec_data['skeleton_id'] = trajectories_data.index #this is only for the viewer
        rec_data['frame_number'] = trajectories_data.index
        fid.create_table(
                    '/',
                    'trajectories_data',
                    obj=rec_data,
                    filters=TABLE_FILTERS)
        
        fid.create_carray('/', 
                          'skeleton', 
                          tables.Float32Atom(dflt=np.nan), 
                          (tot_rows, 49, 2), 
                          filters=TABLE_FILTERS)
        fid.create_carray('/', 
                          'mask', 
                          tables.Float32Atom(dflt=np.nan), 
                          (tot_rows, roi_size, roi_size), 
                          filters=TABLE_FILTERS)


def _init_partition(skeletons_file, save_dir, roi_size):
    
    base_name = os.path.splitext(os.path.basename(masked_file))[0]
    save_file_g = base_name + '_sample.hdf5'
    save_file_b = base_name + '_sample_bad.hdf5'
    
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    trajectories_data['original_table_row'] = trajectories_data.index
    
    trajectories_data_g = trajectories_data[trajectories_data.is_good_skel]
    trajectories_data_b = trajectories_data[~trajectories_data.is_good_skel]
    
    
    _init_table(save_file_g, trajectories_data_g)
    _init_table(save_file_b, trajectories_data_b)

    output = {'bad':(save_file_b, trajectories_data_b),
            'good':(save_file_g, trajectories_data_g)}
    return output


def save_chuck_f(fname, inds, skel_sc, worm_img_sc):
    with tables.File(fname, 'r+') as fid:
        fid.get_node('/mask')[inds, :,:] = worm_img_sc
        fid.get_node('/skeleton')[inds, :,:] = skel_sc
        


#%%
def get_roi_skel_sample(masked_file, skeletons_file, save_dir, roi_size, chuck_size):
    
    traj_data = _init_partition(skeletons_file, save_dir, roi_size)
    
    for key, data in traj_data.items():
        save_file, trajectories_data = data
        n_batch = mp.cpu_count()
        pool = mp.Pool(n_batch)
        def process_batch(batch):
            for result in pool.map(process_chuck, batch):
                save_chuck_f(save_file, *result)
        
        gen_l = read_data_generator(masked_file, trajectories_data)
        gen_chunk = generate_chunck(gen_l, chuck_size)
        
        batch = []
        for chuck in gen_chunk:
            batch.append(chuck)
            if len(batch) > 10:
                process_batch(batch) 
                batch = []
        process_batch(batch)
    
        add_index(save_file)
#%%
def _get_index(events_tot, val_frac, test_frac):
    inds = np.random.permutation(events_tot)
    test_size = np.round(events_tot*test_frac).astype(np.int)
    val_size = np.round(events_tot*val_frac).astype(np.int)
    
    
    all_ind = {'test' : inds[:test_size], 
               'val': inds[test_size:(val_size+test_size)],
               'train' : inds[(val_size+test_size):]}
    
    return all_ind

def add_index(save_file, val_frac=0.2, test_frac=0.02):
    with tables.File(save_file, 'r+') as fid: 
        if '/index_groups' in fid:
            fid.remove_node('/index_groups')
        
        fid.create_group('/', 'index_groups')
        tot_samples = fid.get_node('/mask').shape[0]
        index = _get_index(tot_samples, val_frac=val_frac, test_frac=test_frac)
        
        for field in index:
            fid.create_carray('/index_groups', 
                          field, 
                          obj = index[field])

#%%
if __name__ == '__main__':
    #read_microns_per_pixel
    masked_file = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/short_movies_new/MaskedVideos/double_pick_260117/HW_worm3_F1-3_Set4_Pos4_Ch3_26012017_153655.hdf5'
    #masked_file = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/short_movies_new/MaskedVideos/double_pick_260117/unc-9_worm10_F1-3_Set1_Pos4_Ch3_26012017_143142.hdf5'
    #masked_file = '/Users/ajaver/OneDrive - Imperial College London/Local_Videos/single_worm/global_sample_v3/egl-43 (n1079)II on food L_2010_07_09__11_04___3___2.hdf5'
    #masked_file = '/Users/ajaver/OneDrive - Imperial College London/Local_Videos/single_worm/global_sample_v3/N2 on food R_2011_03_09__11_58_06___6___3.hdf5'
    skeletons_file = masked_file.replace('.hdf5', '_skeletons.hdf5').replace('MaskedVideos', 'Results')
    
    
    
    save_dir = '/Users/ajaver/Documents/GitHub/work-in-progress/work_in_progress/skeletons_CNN'
    
    roi_size = 128
    chuck_size = 100
    get_roi_skel_sample(masked_file, skeletons_file, save_dir, roi_size, chuck_size)
        
    
#%%
#import matplotlib.pylab as plt
#
#with pd.HDFStore(save_file, 'r') as fid:
#    trajectories_data = fid['/trajectories_data']
#
#ii = 9771
##inds, skel_sc, worm_img_sc = result
#with tables.File(save_file, 'r') as fid:
#    worm_img_sc = fid.get_node('/mask')[ii]
#    skel_sc = fid.get_node('/skeleton')[ii]
#    
#    
#    
#    plt.figure()
#    plt.imshow(worm_img_sc, interpolation=None, cmap='gray')
#    plt.plot(skel_sc[:, 0], skel_sc[:, 1], '.-')
#    plt.grid('off')
