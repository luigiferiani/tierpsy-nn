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

from tierpsy.helper.params import read_fps
from tierpsy_features.smooth import get_group_borders, SmoothedWorm

sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
from misc import get_rig_experiments_df

#import pymysql
#import pandas as pd
#import numpy as np
#import traceback
#import multiprocessing as mp
#import tables
#from tierpsy.analysis.feat_create.obtainFeaturesHelper import WormFromTable

gap_to_interp_seconds = 3
sample_size_frames_s = 90
expected_fps = 25

def _process_row(row):
    #%%
    features_file = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
    fps = read_fps(features_file)
    assert expected_fps == fps
    
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
            
        if np.any(np.isnan(skeletons)):
            
            wormN = SmoothedWorm(skeletons,
                                 gap_to_interp = gap_to_interp
                                 )
            skeletons = wormN.skeleton
        
        borders = get_group_borders(~np.isnan(skeletons[: ,0,0]))
        borders = [x for x in borders if x[1]-x[0]-1 >= sample_size_frames]
        
        yield worm_index, worm_data, skeletons, borders

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

if __name__ == '__main__':
    save_file = '/Users/ajaver/Desktop/CeNDR_skel_smoothed.hdf5'
    
    # pytables filters.
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)
    experiments_df = ini_experiments_df()
    experiments_df = experiments_df[['id', 'strain', 'directory', 'base_name', 'exp_name']]
    with tables.File(save_file, 'w') as tab_fid:
        
        
        #%%
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
        tab_recarray = experiments_df.to_records(index=False)
        tab_recarray = tab_recarray.astype(np.dtype(r_dtype))
        #%%
        tab_fid.create_table(
                    '/',
                    'experiments_data',
                    obj = tab_recarray,
                    filters = TABLE_FILTERS
                    )
        
        
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
        
        tot_skels = 0
        for irow, row in experiments_df.iterrows():
            try:
                features_file = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
                with pd.HDFStore(features_file, 'r') as fid:
                    timestamp_data = fid['/timeseries_data']
            except:
                continue
            for worm_index, worm_data, skeletons, borders in _process_row(row):
                if not borders:
                    continue
                
                for bb in borders:
                    skels = skeletons[bb[0]:bb[1]]
                    assert not np.any(np.isnan(skels))
                    
                    ini_t = worm_data['timestamp'].values[bb[0]]/expected_fps
                    rr = (row['id'],
                          int(worm_index),
                          np.array(row['strain']),
                          ini_t, 
                          tot_skels, 
                          tot_skels + skels.shape[0] - 1
                          )
                    data_table.append([rr])
                    skeletons_data.append(skels)
                    
                    tot_skels += skels.shape[0]
                    
                    print(rr[3:], tot_skels, skeletons_data.shape)
                            
                data_table.flush()
                skeletons_data.flush()
                
                batch_data = []
                print(irow, len(experiments_df))
            break


    with pd.HDFStore(save_file, 'r') as fid:
        skeletons_groups = fid['/skeletons_groups']
    #%%
    ss = skeletons_groups['strain'].unique()
    strains_dict = {x:ii for ii,x in enumerate(np.sort(ss))}
    strains_codes = np.array(list(strains_dict.items()), 
                             np.dtype([('strain', 'S7'), ('strain_id', np.int)]))
    #%%
    with tables.File(save_file, 'r+') as fid:
        fid.create_table(
                    '/',
                    'strains_codes',
                    obj = strains_codes,
                    filters = TABLE_FILTERS
                    )