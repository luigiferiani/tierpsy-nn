#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:52:44 2017

@author: ajaver
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import tables
import random

from tierpsy.helper.params import read_fps
from tierpsy.helper.misc import TimeCounter
from tierpsy_features.smooth import get_group_borders, SmoothedWorm

sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
from misc import get_rig_experiments_df

def _h_divide_in_sets(strain_groups,
                   test_frac = 0.1,
                   val_frac = 0.1
                   ):
    
    
    indexes_per_set = dict(
            test = [],
            val = [],
            train = []
            )
    
    
    for strain_id, dat in strain_groups:
        experiments_id = dat['experiment_id'].unique()
        if len(experiments_id)<=2:
            continue
            
        
        random.shuffle(experiments_id)
        
        tot = len(experiments_id)
        
        train_frac = 1-test_frac-val_frac
        rr = (int(np.ceil(test_frac*tot)),
              int(np.ceil((test_frac+train_frac)*tot))
              )
        
        exp_per_set = dict(
        test = experiments_id[:rr[0]+1],
        val = experiments_id[rr[0]:rr[1]+1],
        train = experiments_id[rr[1]:]
        )
        
        for k, val in exp_per_set.items():
            dd = dat[dat['experiment_id'].isin(exp_per_set[k])].index
            assert len(dd) > 0
            indexes_per_set[k].append(dd)
    
    indexes_per_set = {k:np.concatenate(val) for k,val in indexes_per_set.items()}
    
    return indexes_per_set

def add_sets_index(main_file, val_frac=0.1, test_frac=0.1):
    #%%
    with pd.HDFStore(main_file, 'r') as fid:
            df1 = fid['/skeletons_groups']
            df2 = fid['/strains_codes']
    skeletons_indexes = pd.merge(df1, df2, on='strain')
    # divide data in subsets for training and testing    
    strain_groups = skeletons_indexes.groupby('strain_id')
    
    random.seed(777)
    indexes_per_set = _h_divide_in_sets(strain_groups)
    
    with tables.File(main_file, 'r+') as fid: 
        if '/index_groups' in fid:
            fid.remove_node('/index_groups', recursive=True)
        
        fid.create_group('/', 'index_groups')
        
        for field in indexes_per_set:
            fid.create_carray('/index_groups', 
                          field, 
                          obj = indexes_per_set[field])
    #%%

def _process_file(features_file, fps):
    #%%
    
    sample_size_frames = int(round(90*fps))
    gap_to_interp = int(round(gap_to_interp_seconds*fps))
    
    
    with pd.HDFStore(features_file, 'r') as fid:
        timestamp_data = fid['/timeseries_data']
    
    
    for worm_index, worm_data in timestamp_data.groupby('worm_index'):
        if worm_data.shape[0] < sample_size_frames:
            continue
        
        skel_ids = worm_data.index
        with tables.File(features_file, 'r') as fid:
            skeletons = fid.get_node('/coordinates/skeletons')[skel_ids]
        
        is_bad_skeleton = np.isnan(skeletons[:, 0, 0])
        if np.any(is_bad_skeleton):
            
            wormN = SmoothedWorm(skeletons,
                                 gap_to_interp = gap_to_interp
                                 )
            skeletons = wormN.skeleton
        
        borders = get_group_borders(~np.isnan(skeletons[: ,0,0]))
        borders = [x for x in borders if x[1]-x[0]-1 >= sample_size_frames]
        
        yield worm_index, worm_data, skeletons, is_bad_skeleton, borders

def ini_experiments_df():
    exp_set_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR'
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')

    set_type = 'featuresN'
    
    save_dir = './results_{}'.format(set_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    
    f_ext = '_{}.hdf5'.format(set_type)
    features_files = glob.glob(os.path.join(feats_dir, '**/*{}'.format(f_ext)), recursive=True)
    features_files = [x.replace(f_ext, '') for x in features_files]
    
    experiments_df = get_rig_experiments_df(features_files, csv_files)
    experiments_df = experiments_df.sort_values(by='video_timestamp').reset_index()  
    experiments_df['id'] = experiments_df.index
    return experiments_df

def read_CeNDR_snps():
    fname = '/Users/ajaver/Documents/GitHub/process-rig-data/tests/CeNDR/CeNDR_snps.csv'

    snps = pd.read_csv(fname)
    
    info_cols = snps.columns[:4]
    strain_cols = snps.columns[4:]
    snps_vec = snps[strain_cols].copy()
    snps_vec[snps_vec.isnull()] = 0
    snps_vec = snps_vec.astype(np.int8)
    
    
    snps_c = snps[info_cols].join(snps_vec)
    
    r_dtype = []
    for col in snps_c:
        dat = snps_c[col]
        if dat.dtype == np.dtype('O'):
            n_s = dat.str.len().max()
            dt = np.dtype('S%i' % n_s)
        else:
            dt = dat.dtype
        r_dtype.append((col, dt))
    
    snps_r = snps_c.to_records(index=False).astype(r_dtype)
    return snps_r

if __name__ == '__main__':
    save_file = '/Users/ajaver/Desktop/CeNDR_skel_smoothed.hdf5'
    
    gap_to_interp_seconds = 3
    sample_size_frames_s = 90
    
    # pytables filters.
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)
    experiments_df = ini_experiments_df()
    experiments_df = experiments_df[['id', 'strain', 'directory', 'base_name', 'exp_name']]
    
    experiments_df['fps'] = np.nan
    with tables.File(save_file, 'w') as tab_fid:
        r_dtype = []
        for col in experiments_df:
            dat = experiments_df[col]
            if dat.dtype == np.dtype('O'):
                n_s = dat.str.len().max()
                dt = np.dtype('S%i' % n_s)
            else:
                dt = dat.dtype
            r_dtype.append((col, dt))
        #%%
        table_type = np.dtype([('experiment_id', np.int32),
                               ('worm_index', np.int32),
                          ('strain', 'S10'),
                          ('ini_time_aprox', np.float32),
                          ('ini', np.int32),
                          ('fin', np.int32)
                          ])
            
        data_table = tab_fid.create_table('/',
                                        "skeletons_groups",
                                        table_type,
                                        "Worm feature List",
                                        filters = TABLE_FILTERS)
        
        skeletons_data = tab_fid.create_earray('/', 
                                        'skeletons_data',
                                        atom = tables.Float32Atom(),
                                        shape = (0, 49, 2),
                                        expectedrows = experiments_df.shape[0]*22500,
                                        filters = TABLE_FILTERS)
        
        is_bad_skeleton_data = tab_fid.create_earray('/', 
                                        'is_bad_skeleton',
                                        atom = tables.Int8Atom(),
                                        shape = (0,),
                                        expectedrows = experiments_df.shape[0]*22500,
                                        filters = TABLE_FILTERS)
        
        timer = TimeCounter(tot_frames = len(experiments_df))
        tot_skels = 0
        for irow, row in experiments_df.iterrows():
            try:
                features_file = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
                with pd.HDFStore(features_file, 'r') as fid:
                    timestamp_data = fid['/timeseries_data']
            except:
                continue
            
            features_file = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
            fps = read_fps(features_file)
            experiments_df.loc[irow, 'fps']
            
            for output in _process_file(features_file, fps):
                worm_index, worm_data, skeletons, is_bad_skeleton, borders = output
                
                
                if not borders:
                    continue
                
                for bb in borders:
                    skels = skeletons[bb[0]:bb[1]]
                    assert not np.any(np.isnan(skels))
                    is_bad = is_bad_skeleton[bb[0]:bb[1]]
                    
                    ini_t = worm_data['timestamp'].values[bb[0]]/fps
                    rr = (row['id'],
                          int(worm_index),
                          np.array(row['strain']),
                          ini_t, 
                          tot_skels, 
                          tot_skels + skels.shape[0] - 1
                          )
                    data_table.append([rr])
                    skeletons_data.append(skels)
                    is_bad_skeleton_data.append(is_bad)
                    
                    tot_skels += skels.shape[0]
                    
                    #print(rr[3:], tot_skels, skeletons_data.shape)
                            
                data_table.flush()
                skeletons_data.flush()
                
                batch_data = []
            
           
            print(timer.get_str(irow+1))
            
        #%%
        #save the experiments table. I do it after the loop to store the fps information
        tab_recarray = experiments_df.to_records(index=False)
        tab_recarray = tab_recarray.astype(np.dtype(r_dtype))
        
        tab_fid.create_table(
                    '/',
                    'experiments_data',
                    obj = tab_recarray,
                    filters = TABLE_FILTERS
                    )
    #%%
    #I am reading the skeletons_group instead of the experiment data, to ignore strains without a valid skeleton
    with pd.HDFStore(save_file, 'r') as fid:
        skeletons_groups = fid['/skeletons_groups']
    #get strain data
    ss = skeletons_groups['strain'].unique()
    strains_dict = {x:ii for ii,x in enumerate(np.sort(ss))}
    strains_codes = np.array(list(strains_dict.items()), 
                             np.dtype([('strain', 'S7'), ('strain_id', np.int)]))
    #%%
    #get snps vector
    snps = read_CeNDR_snps()
    
    with tables.File(save_file, 'r+') as fid:
        if '/strains_codes' in fid:
            fid.remove_node('/strains_codes')
        fid.create_table(
                    '/',
                    'strains_codes',
                    obj = strains_codes,
                    filters = TABLE_FILTERS
                    )
        fid.create_table(
                    '/',
                    'snps_data',
                    obj = snps,
                    filters = TABLE_FILTERS
                    )
    #%%
    add_sets_index(save_file, val_frac=0.1, test_frac=0.1)