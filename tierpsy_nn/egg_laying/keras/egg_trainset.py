#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:54:33 2017

@author: ajaver
"""
import h5py
import tables
import pandas as pd
import numpy as np
import pymysql
import os
import matplotlib.pylab as plt
import cv2

from tierpsy.analysis.ske_create.helperIterROI import getROIfromInd
from tierpsy.analysis.ske_init.filterTrajectModel import shift_and_normalize

def plot_seq(seq_worm, irow=0, n_rows=1):
    seq_size = seq_worm.shape[0]
    for ii, img in enumerate(seq_worm):
        nn = ii+1 + seq_size*irow
        plt.subplot(n_rows, seq_size, nn)
        plt.imshow(img, interpolation='none', cmap='gray')
        plt.axis('off')

def read_egg_events(fname = 'manually_verified.xlsx'):
    egg_lists = pd.read_excel(fname)
    all_eggs = []
    for ii, row in egg_lists.iterrows():
        row = row.dropna()
        row = row.values
        
        base_name =  row[0].replace('.hdf5', '')
        egg_frames = list(map(int, row[1:]))
        
        if len(egg_frames) > 0:
            all_eggs += zip([base_name]*len(egg_frames), egg_frames)
        
    egg_events = pd.DataFrame(all_eggs, columns=['base_name', 'frame_number'])
    return egg_events

def get_files(cur, base_name):
    sql = '''
    select results_dir
    from experiments
    where base_name ="{}"
    '''.format(base_name)
            
    cur.execute(sql)
    results_dir, = cur.fetchone()
    masked_file = os.path.join(results_dir, base_name + '.hdf5')
    skel_file = os.path.join(results_dir, base_name + '_skeletons.hdf5')
    
    return masked_file, skel_file
#%%
def get_index(events_tot, val_frac, test_frac):
    rand_seed = 1337
    np.random.seed(rand_seed)  # for reproducibility
    
    inds = np.random.permutation(events_tot)
    test_size = np.round(events_tot*test_frac).astype(np.int)
    val_size = np.round(events_tot*val_frac).astype(np.int)
    
    
    all_ind = {'test' : inds[:test_size], 
               'val': inds[test_size:(val_size+test_size)],
               'train' : inds[(val_size+test_size):]}
    
    return all_ind

def randomize_by_event(egg_event_id, val_frac, test_frac):
    events_tot = egg_event_id.size
    
    #randomize by individual event
    uevents = np.unique(egg_event_id)
    dd = get_index(uevents.size,  val_frac, test_frac)
    event_set = {x:uevents[val] for x,val in dd.items()}
    
    #assing a number to each flag, just to ease calculation
    field_flag = {x:ii for ii, x in enumerate(event_set)}
    
    #map each event to a group
    event_map = {}
    for ff, val in event_set.items():
        for x in val:
            event_map[x] = field_flag[ff]
    
    #finally assign each row to the corresponding group
    event_group_vec = np.array([event_map[x] for x in egg_event_id])
    indexes = np.arange(events_tot)
    all_ind = {}
    for field in field_flag:
        good = event_group_vec == field_flag[field]
        all_ind[field] = indexes[good]
    
    return all_ind
#%%
def add_train_indexes(training_file, val_frac = 0.1, test_frac = 0.1):
    
    with tables.File(training_file, 'r') as fid:
        egg_event_id = fid.get_node('/egg_event_id')[:]
    
    all_ind = randomize_by_event(egg_event_id, val_frac, test_frac)
    with tables.File(training_file, 'r+') as fid:
        if '/partitions' in fid:
            fid.remove_node('/partitions', recursive=True)
        
            
        grp = fid.create_group('/', 'partitions')
        for field, indexes in all_ind.items():
            fid.create_array(grp, field, obj=indexes)
#%%
if __name__ == '__main__':
    training_file = 'samples_eggs_resized_seq.hdf5'
    roi_size = 128
    roi_fixed = -1
    win_d = 2
    win_size = 2*win_d + 1
    win_offset = (win_d*2+2)
    
    _debug = False
    
    conn = pymysql.connect(host='localhost', db = 'single_worm_db')
    cur = conn.cursor()
    
    egg_events = read_egg_events()
    tot_events = len(egg_events)
    
    #%%
    
#    tot_samples = 0
#    with h5py.File(training_file, 'w') as fid:
#        egg_events.to_hdf(training_file, 'eggs_data')
#        #egg_events = egg_events[:10]
#        
#        egg_X = fid.create_dataset('/egg_laying_X', 
#                                     (tot_events, win_size, roi_size,roi_size),
#                                     dtype=np.float,
#                                     maxshape=(None, win_size, roi_size,roi_size),
#                                     chunks=(1, win_size, roi_size,roi_size),
#                                     **IMG_FILTERS)
#        egg_Y = fid.create_dataset('/egg_laying_Y', 
#                                     (tot_events,),
#                                     dtype=np.int,
#                                     maxshape=(None,),
#                                     **IMG_FILTERS)
#        
#        event_ids = fid.create_dataset('/egg_event_id', 
#                                     (tot_events,),
#                                     dtype=np.int,
#                                     maxshape=(None,),
#                                     **IMG_FILTERS)
#        
#        vid_group = egg_events.groupby('base_name')
#        tot_vids = len(vid_group)
#        for ivid, (base_name, eggs) in enumerate(vid_group):
#            print(ivid+1, tot_vids, base_name)
#            masked_file, skel_file = _get_files(cur, base_name)
#            
#            with pd.HDFStore(skel_file, 'r') as fid:
#                trajectories_data = fid['/trajectories_data']
#            
#            for irow, egg_frame in eggs['frame_number'].iteritems():
#                g_range = [egg_frame-win_d, egg_frame+win_d]
#                b1_range = [x-win_offset for x in g_range]
#                b2_range = [x+win_offset for x in g_range]
#                
#                ranges = zip((-1,0,1), (b1_range, g_range, b2_range))
#                
#                for lab, ran in ranges:
#                    if lab != 0: 
#                        #ensure there is no good event in the ranges that are suppose to be negative
#                        if any((x>=ran[0] and x<=ran[1]) for x in eggs['frame_number'].values):
#                            continue
#                
#                    worm_rois = np.zeros((win_size, roi_size, roi_size), dtype=np.float)
#                    worm_rois_d = np.zeros((win_size-1, roi_size, roi_size), dtype=np.float)
#                    prev_img = None
#                    
#                    empty_array = True
#                    for ii, frame_number in enumerate(range(ran[0], ran[1]+1)):
#                        output = getROIfromInd(masked_file, trajectories_data, frame_number, 1, roi_size=roi_fixed)
#                        if output is not None:
#                            row, worm_roi, roi_corner = output
#                            
#                            worm_roi = worm_roi.astype(np.float)
#                            worm_roi = shift_and_normalize(worm_roi)
#                            if worm_roi.shape[0] != roi_size:
#                                worm_roi = cv2.resize(worm_roi, (roi_size,roi_size))
#                            worm_rois[ii] = worm_roi
#                            empty_array = False
#                            
#                            if prev_img is not None:
#                                mask = (worm_roi*prev_img) != 0
#                                worm_diff = np.zeros_like(worm_roi)
#                                worm_diff[mask] = worm_roi[mask] - prev_img[mask]
#                                worm_rois_d[ii-1] = worm_diff
#                            prev_img = worm_roi
#                        
#                        
#                            
#                    if not empty_array:
#                        #worm_rois_n = shift_and_normalize(worm_rois)
#                        egg_X[tot_samples] = worm_rois
#                        egg_Y[tot_samples] = lab
#                        event_ids[tot_samples] = irow
#                        
#                        egg_X_diff[tot_samples] = worm_rois_d
#                        
#                        tot_samples += 1
#                        if event_ids.size <= tot_samples:
#                            for datset in [egg_X, egg_Y, event_ids, egg_X_diff]:
#                                datset.resize(tot_samples+ 100, axis=0)                
#        
#        #close and add a randomized training set
#        for datset in [egg_X, egg_Y, event_ids, egg_X_diff]:
#            datset.resize(tot_samples, axis=0)
#    
#    add_train_indexes(training_file)