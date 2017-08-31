#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:40:14 2016

@author: ajaver
"""

import tables
import numpy as np
import pandas as pd
#from sklearn import cross_validation


if __name__ == '__main__':
    all_samples_file = '/Users/ajaver/OneDrive - Imperial College London/training_data/worm_ROI_samplesI.hdf5'


    with pd.HDFStore(all_samples_file) as fid:
        sample_data = fid['/sample_data']

    valid_data = sample_data.loc[sample_data['label_id']>0]
    print(valid_data['label_id'].value_counts())

    label_id = valid_data['label_id'].values

    with tables.File(all_samples_file, 'r') as fid:
        d = (40, 120)
        
        masks = fid.get_node('/', 'mask')[valid_data.index, d[0]:d[1], d[0]:d[1]]
        full_data = fid.get_node('/', 'full_data')[valid_data.index, d[0]:d[1], d[0]:d[1]]
        
    #%%
    tot_samples = full_data.shape[0]

    all_ind = np.random.permutation(tot_samples)

    test_size = 1000
    val_size = 1000

    all_ind = {'test' : all_ind[:test_size], 
               'val': all_ind[test_size:(val_size+test_size)],
               'train' : all_ind[(val_size+test_size):]}
       
    #%%
    table_filters = tables.Filters(complevel=5, 
                                   complib='zlib', 
                                   shuffle=True, 
                                   fletcher32=True)

    with tables.File('sample.hdf5', 'w') as fid:
        fid.create_table('/', 
                         'sample_data', 
                         obj=valid_data.to_records(), 
                         filters=table_filters)
        
        for field, ind in all_ind.items():
            fid.create_earray('/',
                     field + '_y', 
                     obj = label_id[ind],
                     filters=table_filters)
            
            fid.create_earray('/',
                     field + '_x', 
                     obj = masks[ind],
                     filters=table_filters)
            
            fid.create_earray('/',
                     field + '_full_x', 
                     obj = full_data[ind],
                     filters=table_filters)
            

    #%%

    #sample_data = [(masks[x], is_worms[x]) for x in [train_ind, val_ind, test_ind]]
    #
    #
    #with gzip.open( "sample.pkl.gz", "wb" ) as fid:
    #    pickle.dump( sample_data,  fid)
    ##%%
    #with gzip.open( "sample.npz.gz", "wb" ) as fid:
    #    pickle.dump( sum(map(list, sample_data), []),  fid)
    #
    ##%%
    #import gzip
    #import pickle
    #ff = '/Users/ajaver/OneDrive - Imperial College London/training_data/sample.pkl.gz'
    #with gzip.open(ff, "rb" ) as fid:
    #    sample_data = pickle.load(fid)
    #
    #
    ##%%
    #import tables
    #field_names = ['train_x', 'train_y', 
    #             'val_x', 'val_y',
    #             'test_x', 'test_y']
    #sample_data_l = sum(map(list, sample_data), [])
    #table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
    #    
    #with tables.File('sample.hdf5', 'w') as fid:
    #    for field, dat in zip(field_names, sample_data_l):
    #        fid.create_earray('/',
    #                 field, 
    #                 obj = dat,
    #                 filters=table_filters)