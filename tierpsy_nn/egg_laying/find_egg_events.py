#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:11:22 2017

@author: ajaver
"""
import os
import cv2
import pandas as pd
import numpy as np
import pymysql
import matplotlib.pylab as plt
#import multiprocessing as mp

from egg_augmentation import normalize_seq
from modified_mobilenet import load_saved_model, MobileNetE

from tierpsy.analysis.ske_create.helperIterROI import generateMoviesROI
from tierpsy.analysis.ske_create.helperIterROI import getROIfromInd


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
#%%
def get_files(base_name):
    
    conn = pymysql.connect(host='localhost', db = 'single_worm_db')
    cur = conn.cursor()
    
    sql = '''
    select results_dir
    from experiments
    where base_name ="{}"
    '''.format(base_name)
            
    cur.execute(sql)
    results_dir, = cur.fetchone()
    masked_file = os.path.join(results_dir, base_name + '.hdf5')
    skel_file = os.path.join(results_dir, base_name + '_skeletons.hdf5')
    
    conn.close()
    return masked_file, skel_file

#%%
def get_possible_eggs(worm_probs, thresh ):
    inds = np.where(worm_probs[:,1]>thresh)[0]
    inds = inds[np.argsort(worm_probs[inds,1])][::-1]
    return inds

def plot_probs(worm_probs, thresh= 0.99, maker='v', col='b'):
    inds = get_possible_eggs(worm_probs, thresh)
    plt.plot(worm_probs[:,1], col)
    plt.plot(inds, worm_probs[inds,1], maker+col)
    
    return inds

def read_seq(masked_file, trajectories_data, ind, roi_size=-1):
    worm_seq = []
    for frame_number in range(ind-2, ind+3):
        output = getROIfromInd(masked_file, trajectories_data, frame_number, 1, roi_size)
        if output is not None:
            row, worm_roi, roi_corner = output
            worm_seq.append(worm_roi)
    return worm_seq

def plot_indexes(inds, roi_size=-1, n_rows = 5):

    
    for iseq, ind in enumerate(inds):
        worm_seq = read_seq(ind, roi_size)
    #for iseq, worm_seq in enumerate(worm_seqs[:5]):
        
        if iseq % n_rows == 0:
            plt.figure()
        
        irow = iseq % n_rows
        
        seq_size = len(worm_seq)
        #seq_size = worm_seq.shape[-1]
        
        for ii in range(seq_size):
            nn = ii+1 + seq_size*irow
            plt.subplot(n_rows, seq_size, nn)
            plt.imshow(worm_seq[ii], interpolation = 'none', cmap='gray')
            #plt.imshow(worm_seq[:, :, ii], interpolation = 'none', cmap='gray')
            plt.axis('off')
def _fix_padding(worm_img, roi_corner, roi_size):        
    if worm_img.shape[0] != roi_size:
        dd = int(roi_size - worm_img.shape[0])
        assert dd > 0
        if roi_corner[1] == 0:
            worm_img = np.pad(worm_img, ((dd, 0),(0,0)), mode='constant')
        else:
            worm_img = np.pad(worm_img, ((0, dd),(0,0)), mode='constant')
        
    
    if worm_img.shape[1] != roi_size:
        dd = int(roi_size - worm_img.shape[1])
        if roi_corner[0] == 0:
            worm_img = np.pad(worm_img, ((0,0),(dd, 0)), mode='constant')
        else:
            worm_img = np.pad(worm_img, ((0,0),(0, dd)), mode='constant')
    
    if not all(x==roi_size for x in worm_img.shape):
        import pdb
        pdb.set_trace()
    return worm_img

def get_egg_probabilities(masked_file, trajectories_data, model, roi_size = -1, progress_prefix = ''):
    #%%
    #%%
    tot_frames = trajectories_data['frame_number'].max() + 1
    progress_prefix = progress_prefix + ' Searching egg events'
    ROIs_generator = generateMoviesROI(masked_file, 
                                       trajectories_data,
                                       roi_size =roi_size,
                                       progress_prefix = progress_prefix)
    roi_model = model.input_shape[1]
    buff_size = model.input_shape[-1]
    output_size = model.output_shape[-2]
    assert model.output_shape[-1] == 2
    
    worm_buff = []
    seq_dat = []
    worm_probs = np.full((tot_frames, output_size,  2), np.nan)
    
    for worms_in_frame in ROIs_generator:
        assert len(worms_in_frame) == 1 #we are only dealing with one worm case
        for ind, (worm_img, roi_corner) in worms_in_frame.items():
            row_data = trajectories_data.loc[ind]
            frame_number = row_data['frame_number']
            worm_img = _fix_padding(worm_img, roi_corner, row_data['roi_size'])
            
            if len(worm_buff) < buff_size:
                worm_buff.append(worm_img)
            else:
                worm_buff = worm_buff[1:] + [worm_img]
                worm_seq = np.array(worm_buff, np.float32)
                worm_seq = normalize_seq(worm_seq, channel_axis=0)
                if worm_img.shape[0] != roi_model:
                    worm_seq = [cv2.resize(x, (roi_model,roi_model)) for x in worm_seq]
                
                worm_seq = np.rollaxis(np.array(worm_seq), 0, 3)
                
                #worm_seq = shift_and_normalize(worm_seq)
                seq_dat.append((frame_number-1, worm_seq))
                
    
        if (len(seq_dat)+1) % 100 == 0:
            frame_numbers, worm_seq_batch = map(np.array, zip(*seq_dat))
            
            worm_prob_batch = model.predict(worm_seq_batch, verbose=0)
            worm_probs[frame_numbers] = worm_prob_batch
            seq_dat = []
            #return worm_probs
    
    if len(seq_dat) > 0:
        frame_numbers, worm_seq_batch = map(np.array, zip(*seq_dat))
        
        worm_prob = model.predict(worm_seq_batch, verbose=0)
        worm_probs[frame_numbers] = worm_prob
    
    return worm_probs

def process_data(base_name, eggs, save_results_dir, model):
    
    #%%   
    
    
    masked_file, skel_file = get_files(base_name)
    
    
    with pd.HDFStore(skel_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    #trajectories_data = trajectories_data[trajectories_data['frame_number']< 1000]
    #%%
    Y_pred = get_egg_probabilities(masked_file, 
                                   trajectories_data, 
                                   model=model, 
                                   roi_size = -1,
                                   progress_prefix = base_name + ' model resized -> ' )
    
    
    Y_true = np.zeros(Y_pred.shape[0])
    egg_events = eggs['frame_number'].values
    egg_events = egg_events[egg_events<Y_pred.shape[0]]
    
    Y_true[egg_events] = 1
    
    sname = os.path.join(save_results_dir, base_name + '_eggs')
    np.savez(sname, Y_pred, Y_true)

if __name__ == '__main__':
    #%%
    #save_results_dir = './results'
    save_results_dir = '/Users/ajaver/OneDrive - Imperial College London/egg_laying/results_N'
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    
    
    
    #%%
    #model_trained_path = 'model_egg_laying3_epocs20_20170308_194818.h5'
    #model_trained_path = 'model_egg_laying_diff_20170309_193209.h5'
    #model_paths = '/Volumes/behavgenom_archive$/Avelino/neural_networks/eggs_tests/logs/main_20170328_180144/'
    
    
    egg_events = read_egg_events()
    
    #process only files that has not been finished
    files_done = [x.partition('_eggs.')[0] for x in os.listdir(save_results_dir) if '_eggs' in x]
    
    idone = egg_events.base_name.isin(files_done)
    egg_events = egg_events[~idone]
    
    
    
    vid_group = egg_events.groupby('base_name')
    tot = len(vid_group)
    
    #%%
    #model_path_resized = os.path.join(model_paths, 'main_resized-008-0.0891.h5')
    #model_path = 'egg_mobilenet-00649-0.3214.h5'
    model_path = 'egg_mobilenet_W5-2-2_R128_B32-00349-0.3753.h5'
    
    try:
        model_e = load_saved_model(model_path)
    except:
        model_e = MobileNetE(rows=128, 
              cols=128,
              win_size=5,
              y_offset_left=2,
              y_offset_right=2,
              nb_classes=2
              )
        model_e.load_weights(model_path)
    
    
    results = []
    for ii, (base_name, eggs) in enumerate(vid_group):
        print('{} of {}'.format(ii, tot))
        try:
            process_data(base_name, eggs, save_results_dir, model_e)
        except:
            pass
