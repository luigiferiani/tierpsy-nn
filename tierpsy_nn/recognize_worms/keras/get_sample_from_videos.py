#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:01:03 2016

@author: ajaver
"""
import pandas as pd
import tables
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm
# from MWTracker.trackWorms.getSkeletonsTables import getWormROI
from tierpsy.analysis.ske_create.helperIterROI import getWormROI
#%%
def pad_if_necessary(worm_roi, roi_corner, roi_size):

    #if the dimenssion are correct return
    if worm_roi.shape == (roi_size, roi_size):
        return worm_roi, roi_corner

    #corner dimensions are inverted to be the same as in the skeletons
    roi_corner = roi_corner[::-1]
    def get_corner_and_range(curr_dim):
        curr_corner = roi_corner[curr_dim]
        curr_shape = worm_roi.shape[curr_dim]
        #get the shifted corner and start of the roi
        if curr_corner == 0:
            new_corner = curr_shape - roi_size
            new_range = (np.abs(new_corner), roi_size)
        else:
            new_corner = curr_corner
            new_range = (0, curr_shape)

        return new_corner, new_range


    new_corner, new_ranges = zip(*[get_corner_and_range(x) for x in range(2)])
    #corner dimensions are inverted to be the same as in the skeletons
    new_corner = new_corner[::-1]

    new_worm_roi = np.zeros((roi_size, roi_size), dtype = worm_roi.dtype)
    new_worm_roi[new_ranges[0][0]:new_ranges[0][1],
                 new_ranges[1][0]:new_ranges[1][1]] = worm_roi
    return new_worm_roi, new_corner



#%%
def get_all_img_ROI(img, frame_data, roi_size=-1):
    #more generic function that tolerates different ROI size
    worms_in_frame = {}
    for irow, row in frame_data.iterrows():
        worm_roi = roi_size if roi_size > 0 else row['roi_size']

        worms_in_frame[irow] = getWormROI(img, row['coord_x'], row['coord_y'], worm_roi)
    return worms_in_frame
#%%
def get_ROI_fix_size(worms_in_frame, roi_size):
    tot_worms = len(worms_in_frame)
    worm_imgs = np.zeros((tot_worms, roi_size, roi_size))
    roi_corners = np.zeros((tot_worms,2), np.int)
    indexes = np.array(sorted(worms_in_frame.keys()))

    for ii, irow in enumerate(indexes):
        worm_img, roi_corner = worms_in_frame[irow]
        worm_img, roi_corner = pad_if_necessary(worm_img, roi_corner, roi_size)

        worm_imgs[ii] = worm_img
        roi_corners[ii, :] = roi_corner
    return indexes, worm_imgs, roi_corners

def get_traj_data(masked_file, skeletons_file, use_full_frames, sample_frame_interval=-1):
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']

    with tables.File(masked_file, 'r') as fid:
        tot_frames = fid.get_node('/mask').shape[0]

        if use_full_frames or sample_frame_interval < 0:
            sample_frame_interval = fid.get_node('/full_data')._v_attrs['save_interval']


    useful_columns = ['frame_number', 'worm_index_joined', 'skeleton_id',
       'coord_x', 'coord_y', 'threshold', 'has_skeleton', 'roi_size', 'area',
       'timestamp_raw', 'timestamp_time', 'is_good_skel']

    sample_frames = range(0, tot_frames, sample_frame_interval)
    good_rows = trajectories_data['frame_number'].isin(sample_frames)

    sample_traj_data = trajectories_data.loc[good_rows, useful_columns]
    sample_traj_data['img_row_id'] = np.arange(len(sample_traj_data), dtype=np.int)

    assert (sample_traj_data.index == sample_traj_data['skeleton_id']).all()
    sample_traj_data.index = sample_traj_data['img_row_id']

    sample_traj_data['has_skeleton'] = sample_traj_data['has_skeleton'].astype(np.uint8)
    sample_traj_data['is_good_skel'] = sample_traj_data['is_good_skel'].astype(np.uint8)


    return sample_traj_data


def read_group_images(fid, img_h5path, sample_traj_data, roi_size):
    traj_group_by_frame = sample_traj_data.groupby('frame_number')
    tot_samples = len(sample_traj_data)

    sample_rois = np.zeros((tot_samples, roi_size, roi_size))
    sample_corners = np.zeros((tot_samples, 2))

    img_data = fid.get_node(img_h5path)
    is_full_dat = img_h5path == '/full_data'
    for ii, (current_frame, frame_data) in enumerate(traj_group_by_frame):
        img_ind = ii if is_full_dat else current_frame

        img = img_data[img_ind]
        worms_in_frame = get_all_img_ROI(img, frame_data, roi_size)
        indexes, worm_imgs, roi_corners = get_ROI_fix_size(worms_in_frame, roi_size)

        sample_corners[indexes] = roi_corners
        # sample_rois[indexes] = worms_roi
        sample_rois[indexes] = worm_imgs

    return sample_rois, sample_corners

def read_sample_images(masked_file, sample_traj_data, roi_size, use_full_frames):

    with tables.File(masked_file, 'r') as fid:
        sample_masks, sample_corners = read_group_images(fid,
                                                       '/mask',
                                                       sample_traj_data,
                                                       roi_size)
        if use_full_frames:
            sample_fulls, sample_corners = read_group_images(fid,
                                                          '/full_data',
                                                           sample_traj_data,
                                                           roi_size)
        else:
            sample_fulls = np.zeros([])

    #get sampled skeletons
    sample_traj_data['roi_corner_x'] = sample_corners[:, 0]
    sample_traj_data['roi_corner_y'] = sample_corners[:, 1]

    return sample_masks, sample_fulls


def read_sample_skeletons(skeletons_file, sample_traj_data):
    sample_corners = sample_traj_data[['roi_corner_x', 'roi_corner_y']].values
    with tables.File(skeletons_file, 'r') as fid:
        skeletons_id = sample_traj_data['skeleton_id'].values
        sample_skeletons = fid.get_node('/skeleton')[skeletons_id, :, :] - sample_corners[:, np.newaxis, :]
        sample_cnt1 = fid.get_node('/contour_side1')[skeletons_id, :, :] - sample_corners[:, np.newaxis, :]
        sample_cnt2 = fid.get_node('/contour_side2')[skeletons_id, :, :] - sample_corners[:, np.newaxis, :]
    return sample_skeletons, sample_cnt1, sample_cnt2
#%%
if __name__ == '__main__':

    # all_samples_file = '/Users/lferiani/work_repos/tierpsy-nn/data/worm_ROI_samples.hdf5'
    all_samples_file = '/Users/lferiani/work_repos/tierpsy-nn/data/worm_ROI_samples.hdf5'
    # all_samples_file = '/Users/ajaver/OneDrive - Imperial College London/training_data/worm_ROI_samples.hdf5'
    # root_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/'
    # root_dir = '/Users/lferiani/Desktop/20200107_test'
    # root_dir = '/Volumes/Seagate Bac/SyngentaScreen'
    # root_dir = '/Volumes/behavgenom$/Ida/Data/Hydra/PilotDrugExps'
    # root_dir = '/Volumes/behavgenom$/Ida/Data/Hydra/N2unc25'
    is_save_skeletons = True
    #%%
    # masked_files = glob.glob(os.path.join(root_dir, '**', 'MaskedVideos', '**','*.hdf5'), recursive=True)
    # masked_files = list((Path(root_dir) / 'MaskedVideos/').rglob('*run1*/*.hdf5'))
    masked_files = list(Path(
        '/Volumes/behavgenom$/Ida/Data/Hydra/N2unc25/MaskedVideos/').rglob(
            '*113921*/*.hdf5'))
    masked_files = masked_files + list(Path(
        '/Volumes/behavgenom$/Ida/Data/Hydra/SyngentaScreen/MaskedVideos/20191205').rglob(
            '*prestim*.*.hdf5'))
    print(len(masked_files))
    #%%
    #delete previous data
    dat = [(ii, np.string_(x)) for ii,x in enumerate(masked_files)]
    files_rec = np.core.records.fromrecords(dat, names = ['file_id', 'masked_file'])
    with tables.File(all_samples_file, 'w') as fid:
        fid.create_table('/',
                         'file_list',
                        obj = files_rec)
    #%%
    tot_files = len(masked_files)
    for file_id, masked_file_b in tqdm(files_rec):
        masked_file = masked_file_b.decode('utf-8')
        base_name = os.path.splitext(os.path.basename(masked_file))[0]
        # print('{} of {} {}'.format(file_id+1, tot_files, base_name))

        skeletons_file = masked_file.replace('MaskedVideos', 'Results').replace('.hdf5', '_skeletons.hdf5')
        try:
            with tables.File(skeletons_file, 'r') as fid:
                fid.get_node('/skeleton')
        except:
            print("couldn't get skeletons")
            continue

        roi_size = 160
        use_full_frames = True

        sample_data = get_traj_data(masked_file, skeletons_file, use_full_frames)
        sample_data['file_id'] = file_id

        sample_masks, sample_fulls = read_sample_images(masked_file,
                                                        sample_data,
                                                        roi_size,
                                                        use_full_frames)
        if is_save_skeletons:
            sample_skeletons, sample_cnt1, sample_cnt2 = read_sample_skeletons(
                skeletons_file, sample_data)

        #%%
        dat2save = {'sample_data': sample_data.to_records(index=False),
                    'mask': sample_masks}
        if use_full_frames:
            dat2save['full_data'] = sample_fulls
        if is_save_skeletons:
            dat2save['skeleton'] = sample_skeletons
            dat2save['contour_side1'] = sample_cnt1
            dat2save['contour_side2'] = sample_cnt2

        with tables.File(all_samples_file, 'r+') as fid:
            if not '/sample_data' in fid:
                tab_dtypes = dat2save['sample_data'].dtype

                #create arrays if they do not exists
                table_filters = tables.Filters(
                complevel=5, complib='zlib', shuffle=True, fletcher32=True)

                for field, dat in dat2save.items():
                    if isinstance(dat, np.recarray):
                        save_fun = fid.create_table
                    else:
                        save_fun = fid.create_earray

                    save_fun('/',
                             field,
                             obj = dat,
                             filters=table_filters)

            else:
                #append otherwise
                for field, dat in dat2save.items():
                    tab = fid.get_node('/' + field)
                    #make sure the table has the correct data types
                    dat = dat.astype(tab.dtype)
                    tab.append(dat)
                fid.flush()

#%%


    # %%
    # import matplotlib.pylab as plt
    # for ii, worm_roi in enumerate(sample_masks):
    #     if ii < 50:
    #         continue

    #     plt.figure()
    #     plt.imshow(worm_roi)
    #     if ~np.isnan(sample_skeletons[ii, 0, 0]):
    #         plt.plot(sample_skeletons[ii, :, 0], sample_skeletons[ii, :, 1])
    #         plt.plot(sample_cnt1[ii, :, 0], sample_cnt1[ii, :, 1])
    #         plt.plot(sample_cnt2[ii, :, 0], sample_cnt2[ii, :, 1])

    #     if ii > 80:
    #         break
##




