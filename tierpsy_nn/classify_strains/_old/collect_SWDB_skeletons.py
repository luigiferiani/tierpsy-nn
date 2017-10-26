#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:52:44 2017

@author: ajaver
"""
import pymysql
import pandas as pd
import numpy as np
import traceback
import multiprocessing as mp
import tables
from tierpsy.analysis.feat_create.obtainFeaturesHelper import WormFromTable

from smooth_worm import SmoothedWorm, get_group_borders

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/work-in-progress/work_in_progress/new_features')

if __name__ == '__main__':
    conn = pymysql.connect(host='localhost', database='single_worm_db')
    
    sql = '''
    SELECT *, 
    CONCAT(results_dir, '/', base_name, '_skeletons.hdf5') AS skel_file,
    n_valid_skeletons/(total_time*fps) AS frac_valid
    FROM experiments_full AS e
    JOIN results_summary ON e.id = experiment_id
    WHERE total_time < 905
    AND total_time > 295
    AND strain != '-N/A-'
    AND exit_flag = 'END'
    AND n_valid_skeletons > 120*fps
    ORDER BY frac_valid
    '''
    df = pd.read_sql(sql, con=conn)
    #filter strains that have at least 10 videos
    df = df.groupby('strain').filter(lambda x: len(x) > 10)
    
    
    coord_smooth_window_seconds = 1/3
    gap_to_interp_seconds = 3
    expected_fps = 30
    sample_size_frames = 90*expected_fps 
    
    args = coord_smooth_window_seconds, gap_to_interp_seconds, expected_fps, sample_size_frames
    
    def _process_row(row):
        skeletons_file = row['skel_file']
        fps = row['fps']
        
        worm = WormFromTable(skeletons_file, worm_index = 1)
        worm.correct_schafer_worm()
        
        coords_smooth_window = int(round(fps*coord_smooth_window_seconds))
        gap_to_interp = int(round(fps*gap_to_interp_seconds))
        del_t = fps/expected_fps
        frames_to_interpolate = np.arange(0, 
                                          worm.last_frame-worm.first_frame,
                                          del_t
                                          )
        
        wormN = SmoothedWorm(
                 worm.skeleton,
                 skel_smooth_window = 5,
                 coords_smooth_window = coords_smooth_window,
                 gap_to_interp = gap_to_interp,
                 frames_to_interpolate = frames_to_interpolate
                )
        
        
        borders = get_group_borders(~np.isnan(wormN.skeleton[: ,0,0]))
        borders = [x for x in borders if x[1]-x[0]-1 >= sample_size_frames]
        
        print(skeletons_file)        
        return row, wormN.skeleton, borders
    
    
    def process_file_wrapped(row):
        #https://stackoverflow.com/questions/15314189/python-multiprocessing-pool-hangs-at-join
        try:
            return _process_row(row)
        except:
            print('%s: %s' % (row['skel_file'], traceback.format_exc()))
    
    save_file = '/Users/ajaver/Desktop/SWDB_skel_smoothed_2.hdf5'
    
    # pytables filters.
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)

    with tables.File(save_file, 'w') as tab_fid:
        
        #I need to change the dtype of the strings into 'S' otherwise I cannot save them in pytables
        del df['date']
        tab_recarray = df.to_records(index=False)
        r_dtype = [(x, d[0].name if d[0].name != 'object' else 'S40') for x,d in tab_recarray.dtype.fields.items()]
        tab_recarray = tab_recarray.astype(np.dtype(r_dtype))
        tab_fid.create_table(
                    '/',
                    'experiments_data',
                    obj = tab_recarray,
                    filters = TABLE_FILTERS
                    )
        
        
        table_type = np.dtype([('experiment_id', np.int32),
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
                                        expectedrows = df.shape[0]*15000,
                                        filters = TABLE_FILTERS)
        
        n_batch = 1
        #p = mp.Pool(n_batch)
        
        batch_data = []
        tot_skels = 0
        for irow, row in df.iterrows():
            batch_data.append(row)
            if len(batch_data) == n_batch or irow == df.index[-1]:
                
                results =  list(map(process_file_wrapped, batch_data))
                for row, skeletons, borders in results:
                    if not borders:
                        continue
                    
                    for bb in borders:
                        skels = skeletons[bb[0]:bb[1]]
                        assert not np.any(np.isnan(skels))
                        
                        rr = (row['id'],
                              np.array(row['strain']),
                              bb[0]/expected_fps, 
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
                print(irow, len(df))
        #%%
    
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