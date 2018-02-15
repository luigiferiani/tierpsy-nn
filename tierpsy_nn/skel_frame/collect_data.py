#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:54:06 2018

@author: ajaver
"""

import glob
import os
import tables
import pandas as pd
import numpy as np

def _get_mask_file(feat_file):
    return feat_file.replace('_featuresN.hdf5', '.hdf5').replace('Results', 'MaskedVideos')
    

if __name__ == '__main__':
    main_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR'    
    _save_full = True
    fnames = glob.glob(os.path.join(main_dir, '**', '*_featuresN.hdf5'), recursive=True)
    
    #%%
    e_name =  'skel_examples.hdf5' if _save_full else  'skel_examples_r.hdf5'
    examples_file = os.path.join(main_dir, e_name)
    
    coordinates_fields = {
                        'dorsal_contours':'contour_side2', 
                        'skeletons':'skeleton', 
                        'ventral_contours':'contour_side1'
                    }
    
    feat_file = fnames[1]
    mask_file = _get_mask_file(feat_file)
    
    fid = tables.File(examples_file, 'w')
    fid_feat = tables.File(feat_file, 'r')
    fid_mask = tables.File(mask_file, 'r')
    
    for field in ['full_data', 'mask']:
        node = fid_mask.get_node('/' + field)
        _, im_height, im_width = node.shape
        dataset = fid.create_earray(
                    '/',
                    field,
                    atom = node.atom,
                    shape =(0, *node.shape[1:]),
                    chunkshape = node.chunkshape,
                    filters = node.filters
                    )
        fid.copy_node_attrs(node, dataset)
    
    node = fid_feat.get_node('/coordinates/skeletons')
    
    for field in coordinates_fields.values():
        shape_ = (0, 49, 2)
        fid.create_earray(
                '/',
                field,
                atom = node.atom,
                shape = shape_,
                filters = node.filters
                )
    
    fid.close()
    fid_feat.close()
    fid_mask.close()
    
    #%%
    last_img_index = 0
    last_skel_index = 0
    last_worm_index = 0
    
    all_traj = []
    
    for i_feat, feat_file in enumerate(fnames):
        print(i_feat + 1, len(fnames))
        mask_file = feat_file.replace('_featuresN.hdf5', '.hdf5').replace('Results', 'MaskedVideos')
        
        #%% read data
        with tables.File(mask_file, 'r') as fid:
            node = fid.get_node('/full_data')
            save_interval = node._v_attrs['save_interval']
            
            if _save_full:
                full_frames = node[:]
            
            
            
            node = fid.get_node('/mask')
            
            valid_frames = np.arange(0, node.shape[0], save_interval)
            mask_frames = node[::save_interval]
            
            if _save_full:
                assert mask_frames.shape == full_frames.shape
        
        with pd.HDFStore(feat_file, 'r') as fid:
            trajectories_data = fid['/trajectories_data']    
        
        traj_movie = trajectories_data[trajectories_data['frame_number'].isin(valid_frames)].copy()
        
        
        traj_skels = traj_movie[traj_movie['skeleton_id']>=0]
        skeleton_id = traj_skels['skeleton_id'].values
        
        
        coord_data = {}
        if skeleton_id.size > 0:
            with tables.File(feat_file, 'r') as fid:
                microns_per_pixel = fid.get_node('/trajectories_data')._v_attrs['microns_per_pixel']
                for k in ['skeletons', 'dorsal_contours', 'ventral_contours']:
                    
                    field = coordinates_fields[k]
                    coord_data[field] = fid.get_node('/coordinates/' + k)[skeleton_id, :, :]/microns_per_pixel
                
            #%%save data
        ini_img = last_img_index
        last_img_index = ini_img + mask_frames.shape[0]
        
        v_dict = {k:v for k,v in zip(valid_frames, np.arange(ini_img, last_img_index))}
        
        traj_movie['frame_number'] = traj_movie['frame_number'].map(v_dict)
        
        ini_skel = last_skel_index
        last_skel_index = ini_skel + len(skeleton_id)
        traj_movie.loc[traj_skels.index, 'skeleton_id'] = np.arange(ini_skel, last_skel_index)
        
        ini_w = last_worm_index
        last_worm_index = ini_w + traj_movie.shape[0]
        traj_movie['worm_index_old'] = traj_movie['worm_index_joined']
        traj_movie['worm_index_joined']  = np.arange(ini_w, last_worm_index)
        
        with tables.File(examples_file, 'r+') as fid:
            node = fid.get_node('/mask')
            node.append(mask_frames)
            assert node.shape[0] == last_img_index
            
            if _save_full:
                fid.get_node('/full_data').append(full_frames)
                assert fid.get_node('/full_data').shape[0] == last_img_index
            
            for k in coord_data:
                node = fid.get_node('/' + k)
                node.append(coord_data[k])
                
                assert (node.shape[0] == last_skel_index)
        
        #%%
        good_cols = [x for x in traj_movie.columns if not x in ['timestamp_time', 'old_trajectory_data_index']]
        traj_movie = traj_movie[good_cols]
        
        all_traj.append(traj_movie)
        
        
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)
    
    all_traj = pd.concat(all_traj, ignore_index=True)
    tab_recarray = all_traj.to_records(index=False)
    with tables.File(examples_file, 'r+') as fid:
        newT = fid.create_table(
            '/',
            'trajectories_data',
            obj=tab_recarray,
            filters=TABLE_FILTERS)
    
        