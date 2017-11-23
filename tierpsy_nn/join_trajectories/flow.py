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
import torch
from torch.autograd import Variable

def get_worm_ROI(img, CMx, CMy, roi_size=128):
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

def shift_and_normalize(data):
    '''
    shift worms values by an approximation of the removed background. I used the top95 of the unmasked area. 
    I am assuming region of the background is kept.
    '''
    data_m = data.view(np.ma.MaskedArray)
    data_m.mask = data==0
    if data.ndim == 3:
        sub_d = np.percentile(data, 95, axis=(1,2)) #let's use the 95th as the value of the background
        data_m -= sub_d[:, None, None]
    else:
        sub_d = np.percentile(data, 95)
        data_m -= sub_d
        
    data /= 255
    return data

class ROIFlowBase():
    _current_frame = -1
    _frames2iter = []
    _size = None
    def __init__(self, 
                 mask_file, 
                 feat_file, 
                 roi_size = 128,
                 is_cuda = False):
        self.mask_file = mask_file
        self.feat_file = feat_file
        self.roi_size = roi_size
        self.training = True
        self.is_cuda = is_cuda
        
        with pd.HDFStore(self.feat_file) as fid:
            trajectories_data = fid['trajectories_data']
        trajectories_data = trajectories_data[trajectories_data['skeleton_id']>0]
        
        self.trajectories_data = trajectories_data
        self.group_by_frames = trajectories_data.groupby('frame_number')
        self.group_by_index = trajectories_data.groupby('worm_index_joined')
        
        
        self.fid_masks = tables.File(self.mask_file)
        self.masks = self.fid_masks.get_node('/mask')
        
        self.__iter__()
    
    def _get_frame_rois(self, frame_number):
        frame_data = self.group_by_frames.get_group(frame_number)
        frame_img = self.masks[frame_number]
        worms_in_frame = {}
        
        for irow, row in frame_data.iterrows():
            roi_img, d = get_worm_ROI(frame_img, 
                          row['coord_x'], 
                          row['coord_y'], 
                          self.roi_size
                          )
        
            
            if any(x != self.roi_size for x in roi_img.shape):
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
    
    def __iter__(self):
        self._frames2iter = list(self.group_by_frames.groups.keys())
        if self.training:
            random.shuffle(self._frames2iter)
        
        return self
    
    def __next__(self):
        try:
            self._current_frame = self._frames2iter.pop(0)
            return self._get_frame_pairs(self._current_frame)
        except IndexError:
            raise StopIteration
            
    def __len__(self):
        if self._size is None:
            self._size = len(self.trajectories_data.groups.keys())
        return self._size
    
class ROIFlowBatch(ROIFlowBase):
    def __init__(self, 
                 mask_file, 
                 feat_file, 
                 batch_size = 32,
                 is_torch = True,
                 **argkws):
        super().__init__(mask_file, feat_file, **argkws)
        self.batch_size = batch_size
        self.is_torch = is_torch
        
    def __iter__(self):
        #initialize iterator by frames
        super().__iter__()
        
        
        remainder = []
        while True:
            remainder += super().__next__()
            
            while len(remainder) >= self.batch_size:
                chunk = remainder[:self.batch_size]
                remainder = remainder[self.batch_size:] 
                
                X, Y = zip(*chunk)
                X = [(x1[None, ...],x2[None, ...]) for x1, x2 in X]
                X1, X2 = zip(*X)
                
                Y = np.array(Y)
                X1 = np.concatenate(X1).astype(np.float32)
                X2 = np.concatenate(X2).astype(np.float32)
                
                X1 = shift_and_normalize(X1)
                X2 = shift_and_normalize(X2)
                
                if self.is_torch:
                    X1 = self._to_torch(X1[:, None, ...])
                    X2 = self._to_torch(X2[:, None, ...])
                    Y = self._to_torch(Y.astype(np.int64))
                
                chunk = ((X1, X2), Y)
                yield chunk
    
    def _to_torch(self, dat):
        dat = torch.from_numpy(dat)
        if self.is_cuda:
            dat = dat.cuda()
        dat = Variable(dat)
        return dat
    
    def __len__(self):
        if self._size is None:
            self._size = self.trajectories_data.shape[0]//self.batch_size
        return self._size


#%%
#    import matplotlib.pylab as plt
#    for (d1,d2), y in pairs:
#        plt.figure()
#        plt.subplot(1,2,1)
#        plt.imshow(d1, interpolation='none', cmap='gray')
#        plt.subplot(1,2,2)
#        plt.imshow(d2, interpolation='none', cmap='gray')
#        plt.suptitle(y)
#        
#        
#    for d1,d2 in diff_pairs:
#        plt.figure()
#        plt.subplot(1,2,1)
#        plt.imshow(d1, interpolation='none', cmap='gray')
#        plt.subplot(1,2,2)
#        plt.imshow(d2, interpolation='none', cmap='gray')
#        break
        
    
    
    