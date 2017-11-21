#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:24:09 2017

@author: ajaver
"""
import tables
import pandas as pd
import random
import itertools
import numpy as np

from scipy.ndimage.interpolation import rotate

def getWormROI(img, CMx, CMy, roi_size=128):
    '''
    Extract a square Region Of Interest (ROI)
    img - 2D numpy array containing the data to be extracted
    CMx, CMy - coordinates of the center of the ROI
    roi_size - side size in pixels of the ROI

    -> Used by trajectories2Skeletons
    '''

    if np.isnan(CMx) or np.isnan(CMy):
        return np.zeros(0, dtype=np.uint8), np.array([np.nan] * 2)

    roi_center = int(roi_size) // 2
    roi_range = np.round(np.array([-roi_center, roi_center]))

    # obtain bounding box from the trajectories
    range_x = (CMx + roi_range).astype(np.int)
    range_y = (CMy + roi_range).astype(np.int)

    if range_x[0] < 0:
        range_x[0] = 0
    if range_y[0] < 0:
        range_y[0] = 0
    
    if range_x[1] > img.shape[1]:
        range_x[1] = img.shape[1]
    if range_y[1] > img.shape[0]:
        range_y[1] = img.shape[0]

    worm_img = img[range_y[0]:range_y[1], range_x[0]:range_x[1]]

    roi_corner = np.array([range_x[0], range_y[0]])

    return worm_img, roi_corner


class ROIFlow():
    def __init__(self, mask_file, feat_file, roi_size=128):
        self.mask_file = mask_file
        self.feat_file = feat_file
        self.roi_size = roi_size
        self.training = True
        
        with pd.HDFStore(self.feat_file) as fid:
            trajectories_data = fid['trajectories_data']
        trajectories_data = trajectories_data[trajectories_data['skeleton_id']>0]
        
        self.trajectories_data = trajectories_data
        self.group_by_frames = trajectories_data.groupby('frame_number')
        self.group_by_index = trajectories_data.groupby('worm_index_joined')
        
        
        self.fid_masks = tables.File(self.mask_file)
        self.masks = self.fid_masks.get_node('/mask')
        
    
    def _get_frame_rois(self, frame_number):
        frame_data = self.group_by_frames.get_group(frame_number)
        frame_img = self.masks[frame_number]
        worms_in_frame = {}
        
        for irow, row in frame_data.iterrows():
            roi_img, d = getWormROI(frame_img, 
                          row['coord_x'], 
                          row['coord_y'], 
                          self.roi_size
                          )
        
            
            if roi_img.size == 0:
                continue
            
            worms_in_frame[row['worm_index_joined']] = roi_img
            
        return worms_in_frame
    
    def _get_frame_pairs(self, frame_number):
        frame_rois = self._get_frame_rois(frame_number)
        
        w_inds = list(frame_rois.keys())
        diff_pairs = []
        for i1, i2 in list(itertools.combinations(w_inds, 2)):
            dd = (frame_rois[i1], frame_rois[i2])
            dd = [self._transform(x) for x in dd]
            diff_pairs.append(dd)
        
        same_pairs = []
        for i1 in w_inds:
            w_data = self.group_by_index.get_group(i1)
            times = list(w_data['frame_number'].values)
            times.remove(frame_number)
            
            tr = random.choice(times)
            frame2_rois= self._get_frame_rois(tr)
            
            for w2 in w_inds:
                if w2 in frame2_rois:
                    dd = (frame_rois[w2], frame2_rois[w2])
                    dd = [self._transform(x) for x in dd]
                    same_pairs.append(dd)
        
        pairs = [(x, 0) for x in same_pairs] + [(x, 1) for x in diff_pairs]
        random.shuffle(pairs)
        return pairs

    def _transform(self, img):
        #random rotation
        if self.training:
            ang = random.uniform(-180, 180)
        return rotate(img, ang, reshape=False)


if __name__ == '__main__':
    mask_file = '/Users/ajaver/OneDrive - Imperial College London/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5')
        
    g = ROIFlow(mask_file, feat_file)
    
    time_frames = list(g.group_by_frames.groups.keys())
    random.shuffle(time_frames)
    frac = int(len(time_frames)*0.8)
    train_frames, test_frames = time_frames[:frac], time_frames[frac:]
    
    frame_number = random.choice(train_frames)
    pairs = g._get_frame_pairs(frame_number)
    
    #%%
    
    #%%
    import matplotlib.pylab as plt
    for (d1,d2), y in pairs:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(d1, interpolation='none', cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(d2, interpolation='none', cmap='gray')
        plt.suptitle(y)
        
#        
#    for d1,d2 in diff_pairs:
#        plt.figure()
#        plt.subplot(1,2,1)
#        plt.imshow(d1, interpolation='none', cmap='gray')
#        plt.subplot(1,2,2)
#        plt.imshow(d2, interpolation='none', cmap='gray')
#        break
        
    
    
    