#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:41:59 2020

@author: lferiani
"""

import h5py
import tables
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile

# names
dataset_fname = Path.home()/'work_repos/tierpsy-nn/data/worm_ROI_samples.hdf5'
annotators = ['Priota', 'Ziwei']
out_fnames = [dataset_fname.parent / (dataset_fname.stem + '_{}'.format(c))
              for c in annotators]
out_fnames = [f.with_suffix('.hdf5') for f in out_fnames]

# read untouched data
with pd.HDFStore(dataset_fname) as fid:
    sample_data = fid['/sample_data']

# add column for shuffled showing data
if 'resampled_index' not in sample_data:
    sample_data['resampled_index'] = np.random.permutation(len(sample_data))

sample_data['img_row_id'] = sample_data.index
sample_data.index = sample_data['resampled_index'].values

if 'label_id' not in sample_data:
    sample_data['label_id'] = 0


# write
n_annotators = len(annotators)
n_common_rois = 1000
n_rois = sample_data.shape[0]
n_unique_rois_each = int(np.round((n_rois - n_common_rois) / n_annotators))

# ac is annotator counter
for ac, out_fname in enumerate(out_fnames):

    # first, copy everything over
    print('copying')
    # with h5py.File(dataset_fname, 'r') as fid:
    #     with h5py.File(out_fname, 'w') as fidout:
    #         # fidout.create_group('/')
    #         for dataset in tqdm(fid.keys()):
    #             fid.copy(dataset, fidout['/'])
    copyfile(dataset_fname, out_fname)

    # create sample_data for this particular one
    # find bounds
    lower = n_common_rois + ac * n_unique_rois_each
    upper = n_common_rois + (ac+1) * n_unique_rois_each
    print(lower, upper)
    if out_fname == out_fnames[-1]:
        upper = n_rois
    print(lower, upper)
    # create logical index
    idx_tocopy = ((sample_data['resampled_index'] >= lower) &
                  (sample_data['resampled_index'] < upper))
    idx_tocopy = idx_tocopy | (sample_data['resampled_index'] < n_common_rois)
    # get copy
    sd_tocopy = sample_data[idx_tocopy].copy()
    sd_tocopy.reset_index(drop=True, inplace=True)

    # now checks:
    print(sd_tocopy['resampled_index'].min(),
          sd_tocopy['resampled_index'].max(),
          sd_tocopy['resampled_index'].shape,
          )
    assert (sd_tocopy['resampled_index'].shape[0]
            == upper-lower+n_common_rois)

    sd_tocopy['resampled_index_original'] = sd_tocopy['resampled_index'].copy()
    idx = sd_tocopy['resampled_index'] >= n_common_rois
    sd_tocopy.loc[idx, 'resampled_index'] = (sd_tocopy['resampled_index'][idx]
                                             - lower + n_common_rois)

    print(sd_tocopy['resampled_index'].min(),
          sd_tocopy['resampled_index'].max(),
          sd_tocopy['resampled_index'].shape,
          )
    assert (sd_tocopy['resampled_index'].shape[0]
            == upper-lower+n_common_rois)

    print(' ')

    with tables.File(out_fname, "r+") as fid:
        # pytables filters.
        table_filters = tables.Filters(
            complevel=5, complib='zlib', shuffle=True, fletcher32=True)

        newT = fid.create_table(
            '/',
            'sample_data_d',
            obj=sd_tocopy.to_records(index=False),
            filters=table_filters)
        fid.remove_node('/', 'sample_data')
        newT.rename('sample_data')
