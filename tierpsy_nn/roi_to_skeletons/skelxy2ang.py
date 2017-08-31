#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:07:06 2017

@author: ajaver
"""

import tables
import numpy as np
from tierpsy.analysis.ske_create.segWormPython.mainSegworm import resample_curve
#%%
def normalize_skel_angles(angles):
    dangles = np.diff(angles)
    
    #    % need to deal with cases where angle changes discontinuously from -pi
    #    % to pi and pi to -pi.  In these cases, subtract 2pi and add 2pi
    #    % respectively to all remaining points.  This effectively extends the
    #    % range outside the -pi to pi range.  Everything is re-centred later
    #    % when we subtract off the mean.
    #    
    #    % find discontinuities larger than pi (skeleton cannot change direction
    #    % more than pi from one segment to the next)
    positive_jumps = np.where(dangles > np.pi)[0] + 1; #%+1 to cancel shift of diff
    negative_jumps = np.where(dangles <-np.pi)[0] + 1;
    
    #% subtract 2pi from remainging data after positive jumps
    for jump in positive_jumps:
        angles[jump:] = angles[jump:] - 2*np.pi;
    
    #% add 2pi to remaining data after negative jumps
    for jump in negative_jumps:
        angles[jump:] = angles[jump:] + 2*np.pi;
    
    #% rotate skeleton angles so that mean orientation is zero
    mean_angle = np.mean(angles);
    angles = angles - mean_angle;
    
    return angles, mean_angle
    
def transform2skelangles(skeletons_xy):
    skel_resampled = np.array([resample_curve(skel)[0] for skel in skeletons_xy])
    
    delR = np.diff(skel_resampled, axis=1)
    RR = np.linalg.norm(delR, axis=2)
    segment_sizes = np.mean(RR, axis=1)
    
    raw_skel_angles = np.arctan2(delR[:, :,1], delR[:, :,0])
    angle_data = [normalize_skel_angles(x) for x in raw_skel_angles]
    skel_angles, mean_angles = map(np.array, zip(*angle_data))
    
    ini_coord = skel_resampled[:,0,:]
    
    #continous_angle = normalize_skel_angles(mean_angles)[0]

    return skel_angles, mean_angles, segment_sizes, ini_coord

def transform2skelxy(skel_angles, mean_angles, segment_sizes, ini_coord):
    skel_angles_r = skel_angles + mean_angles[:, np.newaxis]
    
    skel_back = np.zeros((skel_angles.shape[0], skel_angles.shape[1]+1, 2))
    skel_back[:, 0, :] = ini_coord
    skel_back[:, 1:, 0] = np.cos(skel_angles_r)*segment_sizes[:, np.newaxis]
    skel_back[:, 1:, 1] = np.sin(skel_angles_r)*segment_sizes[:, np.newaxis]
    skel_back = np.cumsum(skel_back, axis=1)
    return skel_back
#%%
if __name__ == '__main__':
    masked_file = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/short_movies_new/MaskedVideos/double_pick_260117/HW_worm3_F1-3_Set4_Pos4_Ch3_26012017_153655.hdf5'
    #masked_file = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/short_movies_new/MaskedVideos/double_pick_260117/unc-9_worm10_F1-3_Set1_Pos4_Ch3_26012017_143142.hdf5'
    #masked_file = '/Users/ajaver/OneDrive - Imperial College London/Local_Videos/single_worm/global_sample_v3/egl-43 (n1079)II on food L_2010_07_09__11_04___3___2.hdf5'
    #masked_file = '/Users/ajaver/OneDrive - Imperial College London/Local_Videos/single_worm/global_sample_v3/N2 on food R_2011_03_09__11_58_06___6___3.hdf5'
    skeletons_file = masked_file.replace('.hdf5', '_skeletons.hdf5').replace('MaskedVideos', 'Results')
    
    
    
    with tables.File(skeletons_file, 'r') as fid:
        skeletons_xy = fid.get_node('/skeleton')[:1000, : ,: ]
    
    skel_angles, mean_angles, segment_sizes, ini_coord = transform2skelangles(skeletons_xy)
    skel_xy_back = transform2skelxy(skel_angles, mean_angles, segment_sizes, ini_coord)
    #%%
    import matplotlib.pylab as plt
    #%%
    ind = 100
    plt.figure()
    plt.plot(skel_xy_back[ind, :,0], skel_xy_back[ind, :,1], 's-')
    plt.plot(skeletons_xy[ind, :,0], skeletons_xy[ind, :,1], 'o-')
    plt.axis('equal')