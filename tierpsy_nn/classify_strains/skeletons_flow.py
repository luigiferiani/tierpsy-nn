#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 09:20:28 2017

@author: ajaver
"""
import pandas as pd
import tables
import numpy as np
import random

def _h_divide_in_sets(strain_groups,
                   test_frac = 0.1,
                   val_frac = 0.1):
    
    
    indexes_per_set = dict(
            test = [],
            val = [],
            train = [],
            tiny = []
            )
    
    #used for the tiny set
    dd = strain_groups.agg({'strain':'count'})
    top_strains = dd.sort_values(by='strain').index[-5:]
    
    for strain_id, dat in strain_groups:
        experiments_id = dat.experiment_id.unique()
        random.shuffle(experiments_id)
        
        tot = len(experiments_id)
        rr = (int(np.ceil(test_frac*tot)), 
              int(np.ceil((test_frac+val_frac)*tot))
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
    
            
            if strain_id in top_strains and k == 'train': 
                indexes_per_set['tiny'].append(dd[:10])
            
    
    indexes_per_set = {k:np.concatenate(val) for k,val in indexes_per_set.items()}
    
    return indexes_per_set

def _add_index(main_file, val_frac=0.1, test_frac=0.1):
    #%% divide data in subsets for training and testing    
    
    skel_g = SkeletonsFlow(n_batch = 16)
    skeletons, strain_id = skel_g.next_single()
    
    random.seed(777)
    indexes_per_set = _h_divide_in_sets(skel_g.skeletons_groups)
    
    with tables.File(main_file, 'r+') as fid: 
        if '/index_groups' in fid:
            fid.remove_node('/index_groups', recursive=True)
        
        fid.create_group('/', 'index_groups')
        
        for field in indexes_per_set:
            fid.create_carray('/index_groups', 
                          field, 
                          obj = indexes_per_set[field])

    #%%

class SkeletonsFlow():
    def __init__(self,
                n_batch,
                main_file = '/Users/ajaver/Desktop/SWDB_skel_smoothed.hdf5',
                set_type = None,
                min_num_samples = 10,
                valid_strains = None,
                expected_fps = 30,
                sample_size_frames_s = 90,
                sample_frequency_s = 1/10,
                body_range = (8, 41)
                ):
        
        self.n_batch = n_batch
        self.sample_size_frames = sample_size_frames_s*expected_fps
        self.sample_frequency  = sample_frequency_s*expected_fps
        self.main_file = main_file
        self.body_range = body_range
        
        with pd.HDFStore(self.main_file, 'r') as fid:
            df1 = fid['/skeletons_groups']
            df2 = fid['/strains_codes']
        
        #number of classes for the one-hot encoding
        self.n_clases = df2['strain_id'].max() + 1
        
        skeletons_indexes = pd.merge(df1, df2, on='strain')
        good = skeletons_indexes.apply(lambda x : x['fin'] - x['ini'] >= self.sample_size_frames, axis=1)
        skeletons_indexes = skeletons_indexes[good]
        
        if set_type is not None:
            assert set_type in ['train', 'test', 'val', 'tiny']
            with tables.File(self.main_file, 'r') as fid:
                #use previously calculated indexes to divide data in training, validation and test sets
                valid_indices = fid.get_node('/index_groups/' + set_type)[:]
                skeletons_indexes = skeletons_indexes.loc[valid_indices]
        

        skeletons_indexes = skeletons_indexes.groupby('strain_id').filter(lambda x: len(x['experiment_id'].unique()) >= min_num_samples)
        if valid_strains is not None:
            skeletons_indexes = skeletons_indexes[skeletons_indexes['strain'].isin(valid_strains)]


        self.skeletons_indexes = skeletons_indexes
        self.skeletons_groups = skeletons_indexes.groupby('strain_id')
        self.strain_ids = self.skeletons_groups.indices.keys()
        
    def _random_choice(self):
        strain_id, = random.sample(self.strain_ids, 1)
        gg = self.skeletons_groups.get_group(strain_id)
        ind, = random.sample(list(gg.index), 1)
        dat = gg.loc[ind]
        
        r_f = dat['fin'] - self.sample_size_frames
        ini_r = random.randint(dat['ini'], r_f)
        
        row_indices = np.arange(ini_r, ini_r + self.sample_size_frames, self.sample_frequency)
        row_indices = np.round(row_indices).astype(np.int32)
        
        #read data
        with tables.File(self.main_file, 'r') as fid:
            skeletons = fid.get_node('/skeletons_data')[row_indices, :, :]
        
        if np.any(np.isnan(skeletons)):
            print(strain_id, ind, row_indices)
            #this is a bug but i want to solve it fast
            import pdb
            pdb.set_trace()
        
        body_coords = np.mean(skeletons[:, self.body_range[0]:self.body_range[1]+1, :], axis=1)
        skeletons -= body_coords[:, None, :]
        
        return strain_id, skeletons
    
    def _random_transform(self, skeletons):
        #random rotation
        theta = random.uniform(-np.pi, np.pi)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta),  np.cos(theta)]])
        
        skel_r = skeletons.copy()
        for ii in range(skel_r.shape[1]):
            skel_r[:, ii, :] = np.dot(rot_matrix, skeletons[:, ii, :].T).T
        
        #random mirrowing 
        for ii in range(skel_r.shape[-1]):
            skel_r[:, :, ii] *= random.choice([-1, 1])
            
        return skel_r
    
    def next_single(self):
         strain_id, skeletons = self._random_choice()
         X = self._random_transform(skeletons)
         Y = np.zeros(self.n_clases, np.int32)
         Y[strain_id] = 1
         return X,Y
     
    def __next__(self):
        D = [self.next_single() for n in range(self.n_batch)]
        X, Y = map(np.array, zip(*D))
        return X,Y
    
    def __len__(self):
        return self.skeletons_indexes.shape[0]
    
if __name__ == '__main__':
    import sys
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/classify_strains/results'
        main_file = '/work/ajaver/classify_strains/train_set/SWDB_skel_smoothed.hdf5'
    else:        
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains'
        main_file = '/Users/ajaver/Desktop/SWDB_skel_smoothed.hdf5'


    skel_generator = SkeletonsFlow(n_batch = 50, 
                                    main_file = main_file,
                                    set_type='tiny')

    for ii, (X,Y) in enumerate(skel_generator):
        print(ii)
    # X,Y = next(skel_generator)
    
    # import matplotlib.pylab as plt
    
    # for x in X:
        
    #     plt.figure()
    #     plt.subplot(2,1,1)
    #     plt.imshow(x[:, :, 1].T, aspect='auto')
    #     plt.subplot(2,1,2)
    #     plt.imshow(x[:, :, 0].T, aspect='auto')
    #     #%%






    