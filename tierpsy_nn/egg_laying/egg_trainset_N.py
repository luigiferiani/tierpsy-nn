#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:50:32 2017

@author: ajaver
"""
import os
import glob
import tables
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pymysql
from tierpsy.helper.misc import get_base_name


def plot_probs(eggs, worm_probs, true_events, thresh=0.99, plot_log=True):

    def get_possible_eggs(worm_probs, thresh ):
        inds = np.where(worm_probs>thresh)[0]
        inds = inds[np.argsort(worm_probs[inds])][::-1]
        return inds
    
    def _plot_probs(worm_probs, thresh= 0.99, maker='v', col='b'):
        inds = get_possible_eggs(worm_probs, thresh)
        plt.plot(worm_probs, col)
        plt.plot(inds, worm_probs[inds], maker+col)
        
        return inds
    
    
    #eggs, worm_probs_resized, worm_probs_fixed = [results[x] for x in ['eggs', 'worm_probs_resized', 'worm_probs_fixed']]
    
    egg_frames = true_events[true_events>0].index
    
    plt.figure()
    _plot_probs(worm_probs, thresh= thresh, maker='o', col='g')
    plt.plot(egg_frames, np.ones(len(egg_frames)), 'xr')
    plt.title('resized')

def createBlocks(flags_vector, min_block_size=0):
    # divide data into groups of continous indexes
    prev_ind = False
    group_ini = []
    group_fin = []
    for ii, flag_ind in enumerate(flags_vector):

        if not prev_ind and flag_ind:
            group_ini.append(ii)
        if prev_ind and not flag_ind:
            # substract one size this is the condition one after the end of the
            # block
            group_fin.append(ii - 1)
        prev_ind = flag_ind
    # append the last index if the group ended in the last index
    if len(group_ini) - len(group_fin) == 1:
        group_fin.append(ii)
    assert len(group_ini) == len(group_fin)

    # change this into a single list of tuples
    groups = list(zip(group_ini, group_fin))

    # remove any group smaller than the min_block_size
    groups = [gg for gg in groups if gg[1] - gg[0] >= min_block_size]
    return groups

def _get_indexes(inds, true_events, win_size):
    
    tot_frames = true_events.size
    dat = []
    for ii in inds:
        top = ii + win_size + 1
        bot = ii-win_size
        if bot < 0:
            top -= bot #shift top to the right
            bot = 0
            assert top-bot == 2*win_size+1
            
        if top > tot_frames:
            bot -= top-tot_frames
            top = tot_frames
            assert top-bot == 2*win_size+1
        
        x = np.arange(bot, top)
        y = true_events[x].values
        
        assert x.size == y.size
        dat.append((x,y))
    return dat


def get_indexes_for_training(csv_dir, thresh = 0.99, win_size = 5):
    fnames = glob.glob(os.path.join(csv_dir, '*.csv'))

    def _group_by_index(inds):
        gg = []
        n_groups = 0
        curr = np.nan
        for gi in sorted(inds):
            if np.isnan(curr):
                gg.append([gi])
                curr = gi
                n_groups += 1
            elif gi-curr < 3:
                gg[n_groups-1].append(gi)
                curr = gi
            else:
                curr = np.nan
        return gg

    X = []
    Y = []
    BN = []
    for ii, fname in enumerate(fnames):
        bad_inds  = []
        
        base_name = get_base_name(fname)
        print(base_name)
        results = pd.read_csv(fname, index_col=0)
        
        pred_inds = np.where(results['egg_prob']>thresh)[0]
        real_inds = np.where(results['true_events']>0)[0]
        
        dist_d = np.abs(pred_inds-real_inds[:, np.newaxis])
        bad = np.min(dist_d, axis=1)>1
        bad_pred = pred_inds[np.any(dist_d>1, axis=0)]
        if np.any(bad):
            gg = _group_by_index(bad_pred)
            bad_inds = [int(np.median(x)) for x in gg]
            
            
        indexes2add = bad_inds + list(real_inds)
        
        dat = _get_indexes(indexes2add, results['true_events'], win_size)
        xx,yy = zip(*dat)
        
        
        BN += [base_name[:-5]]*len(xx)
        X += xx
        Y += yy
    return BN, X, Y

def save_indexes_for_training(save_name, BN, X_ind, Y):
    
    
    base_names = pd.DataFrame(np.array(BN, dtype='S'), columns=['base_name']).to_records(index=False)
    if os.path.exists(save_name):
        raise FileExistsError(save_name)
    
    with tables.File(save_name, "w") as fid:
        if '/base_names' in fid:
            fid.remove_node('/base_names')
        
        fid.create_table('/',
                        'base_names',
                        obj=base_names
                        )
        
        fid.create_array('/',
                          'X_ind',
                          obj=np.array(X_ind)
                          )
        
        fid.create_array('/',
                          'Y',
                          obj=np.array(Y)
                          )
        
        seq_size = len(X_ind[0])
        tot = len(X_ind)
        fid.create_array('/',
                          'X',
                          atom=tables.UInt8Atom(),
                          shape =(tot, seq_size, 480, 640)
                          )

        

if __name__ == '__main__':
    csv_dir = '/Users/ajaver/OneDrive - Imperial College London/egg_laying/results'
    save_name = '/Users/ajaver/OneDrive - Imperial College London/egg_laying/train_data_eggs.hdf5'
    
    if not os.path.exists(save_name):
        BN, X, Y = get_indexes_for_training(csv_dir)
        save_indexes_for_training(save_name, BN, X, Y)

    with pd.HDFStore(save_name, 'r') as fid:
        base_names = fid['/base_names']
    
    with tables.File(save_name, 'r') as fid:
        X_ind = fid.get_node('/X_ind')[:]
        Y = fid.get_node('/Y')[:]
    #%%
    conn = pymysql.connect(host='localhost', db = 'single_worm_db')
    cur = conn.cursor()
    
    seq_traj_data = []
    for ivid, (base_name, dd) in enumerate(base_names.groupby('base_name')):
        print(ivid, base_name)
        
        
        sql = '''
        select results_dir
        from experiments
        where base_name ="{}"
        '''.format(base_name)
        
        cur.execute(sql)
        results_dir, = cur.fetchone()
        masked_file = os.path.join(results_dir, base_name + '.hdf5')
        inds = dd.index.values
        if False:
            with tables.File(masked_file, 'r') as fid, \
            tables.File(save_name, 'r+') as fid_p:
                X = fid_p.get_node('/X')
                mask = fid.get_node('/mask')
                for ii in inds:
                    seq = mask[X_ind[ii], :, :]
                    X[ii] = seq[None, ...]
        
        
        
        skeletons_file = os.path.join(results_dir, base_name + '_skeletons.hdf5')
        with pd.HDFStore(skeletons_file, 'r') as fid:
            trajectories_data = fid['/trajectories_data']
        #%%
        feats = ['frame_number', 'coord_x', 'coord_y', 'threshold', 'roi_size', 'area']
        trajectories_data.index = trajectories_data['frame_number']
        
        DD = []
        for nn, iseq in enumerate(inds):
            try:
                dat = trajectories_data.loc[X_ind[iseq],feats]
                dat['seq_index'] = iseq
            except KeyError:
                continue
            DD.append(dat)
        
        seq_traj_data += DD
    
    seq_traj_data = pd.concat(seq_traj_data)
    
    #%%
    with tables.File(save_name, "r+") as fid:
        if '/coordinates_data' in fid:
            fid.remove_node('/coordinates_data')
        
        fid.create_table('/',
                        'coordinates_data',
                        obj=seq_traj_data.to_records(index=False)
                        )
    
    #%%
    
    #Add indexes to subdivide in training and validation tests
    with tables.File(save_name, "r") as fid:
        events_tot = fid.get_node('/X').shape[0]
    
    val_frac = 0.05
    rand_seed = 1337
    
    
    np.random.seed(rand_seed)  # for reproducibility
    
    inds = np.random.permutation(events_tot)
    val_size = np.round(events_tot*val_frac).astype(np.int)
    
    all_ind = dict(
                val = inds[:val_size],
                train = inds[val_size:]
               )
    
    inds = np.random.permutation(events_tot)
    
    tiny_set = inds[:300] #select a smaller subset of the images for training
    val_size = np.round(tiny_set.size*0.2).astype(np.int)
    tiny_ind = dict(
                val = tiny_set[:val_size],
                train = tiny_set[val_size:]
               )
    
    
    with tables.File(save_name, "r+") as fid:
        if '/partitions' in fid:
            fid.remove_node('/partitions', recursive=True)
        
            
        grp = fid.create_group('/', 'partitions')
        for field, indexes in all_ind.items():
            fid.create_array(grp, field, obj=indexes)
            
        
        if '/tiny' in fid:
            fid.remove_node('/tiny', recursive=True)
            
        grp = fid.create_group('/', 'tiny')
        for field, indexes in tiny_ind.items():
            fid.create_array(grp, field, obj=indexes)