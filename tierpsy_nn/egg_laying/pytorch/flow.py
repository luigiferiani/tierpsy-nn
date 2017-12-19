#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:50:01 2017

@author: ajaver
"""
import os
import math
import random
import tables
import numpy as np
import pandas as pd
from skimage.transform import resize, rotate
import tqdm

import torch

egg_events_file_dflt = 'egg_events.csv'
local_dir_dlft = '/data/ajaver/egg_laying/training_set'

def _get_random_factors(
        zoom_r = (1, 1.5),
        x_offset_r = (-5, 5),
        y_offset_r = (-5, 5),
        rotation_r = (-np.pi, np.pi),
        b_shift_r = (0., 1.)
        ):
    factors = dict(
            x_offset = random.uniform(*x_offset_r),
            y_offset = random.uniform(*y_offset_r),
            zoom_f = random.uniform(*zoom_r),
            rotation_f =  random.uniform(*rotation_r),
            v_shift = random.choice([False, True]),
            h_shift = random.choice([False, True]),
            b_shift = random.uniform(*b_shift_r),
            )
    return factors
    

def _shift_bgnd(imgs):
    if imgs.size == 0:
        return imgs
    mask = imgs>0
    valid_pix = imgs[mask]
    imgs[~mask] = np.percentile(valid_pix, 95)
        
    return imgs

def _crop_and_augment(snippet, cxx, cyy, 
                      w_roi_size, 
                      roi_output_size, 
                      is_nozoom = True,
                      is_augment=False,
                      is_bgnd_rm = False):
    
    if is_nozoom:
        r = _get_random_factors(zoom_r = (1.,1.), b_shift_r = (0., 0.,))
    elif is_bgnd_rm:
        r = _get_random_factors(b_shift_r = (0., 0.,))
    else:
        r = _get_random_factors()
    
    y_size = snippet.shape[1]
    x_size = snippet.shape[2]
    
    r2 = w_roi_size/2
    r_ini = int(math.floor(r2))
    r_fin = int(math.ceil(r2))
    
    roi_zoomed = int(round(roi_output_size*r['zoom_f']))
    roi_zoomed_half = roi_zoomed//2
    roi_output_half = roi_output_size/2
    
    
    if is_augment:
        cxx += r['x_offset']
        cyy += r['y_offset']
        
    
    snippet_cropped = []
    
    cxx = cxx.astype(np.int32)
    cyy = cyy.astype(np.int32)
    for cx, cy, f in zip(cxx, cyy, snippet):
        x_ini = cx - r_ini
        x_pad_ini = abs(min(0, x_ini))
        
        x_fin = cx + r_fin + 1
        x_pad_fin = max(0, x_fin-x_size)
        
        y_ini = cy - r_ini
        y_pad_ini = abs(min(0, y_ini))
        
        y_fin = cy + r_fin + 1
        y_pad_fin = max(0, y_fin-y_size)
        pad_width = [(y_pad_ini, y_pad_fin), (x_pad_ini, x_pad_fin)]
        
        
        roi = f[y_ini:y_fin, x_ini:x_fin]
        roi = np.pad(roi, 
                     pad_width, 
                     'constant', 
                     constant_values=0)
        
        if is_augment:
            roi[roi<=-0.5] += r['b_shift']  
            
            roi = rotate(roi, r['rotation_f'])
            
            roi = resize(roi, 
                         (roi_zoomed, roi_zoomed), 
                         mode='constant')
            r_ini_z = int(math.floor(roi_zoomed_half - roi_output_half))
            r_fin_z = r_ini_z + roi_output_size
            roi = roi[r_ini_z:r_fin_z, r_ini_z:r_fin_z]
            
            if r['h_shift']:
                roi = roi[::-1, :]
            
            if r['v_shift']:
                roi = roi[:, ::-1]
        else:
            roi = resize(roi, 
                         (roi_output_size, roi_output_size), 
                         mode='constant')
            
        snippet_cropped.append(roi[None, None, ...])
    
    snippet_cropped = np.vstack(snippet_cropped)
    return snippet_cropped
        
    
class EggLayingFlow():
    def __init__(self,
                egg_events_file = egg_events_file_dflt,
                set_type = 'test',
                snippet_size = 5,
                roi_output_s = 96,
                n_batch = 64,
                is_augment = False,
                is_cuda = False,
                is_autoencoder = False,
                is_nozoom = True,
                is_bgnd_rm = False,
                select_near_event = False,
                local_dir = local_dir_dlft
                ):
        
        self.egg_events_file = egg_events_file
        self.set_type =  set_type
        self.snippet_size = snippet_size
        self.roi_output_s = roi_output_s
        self.n_batch = n_batch
        self.is_cuda = is_cuda
        self.is_nozoom = is_nozoom
        self.is_bgnd_rm = is_bgnd_rm
        self.is_autoencoder = is_autoencoder
        self.is_augment = is_augment
        self.local_dir = local_dir
        self.select_near_event = select_near_event
        
        egg_events = pd.read_csv(egg_events_file)
        egg_events = egg_events[egg_events['set_type'] == set_type]
    
        self.egg_g = egg_events.groupby('base_name')
    
    def  _prepare_torch(self, outputs):
        X,Y = map(np.array, zip(*outputs))
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        
        if self.is_cuda:
            X = X.cuda()
            Y = Y.cuda()
            
        X = torch.autograd.Variable(X)
        Y = torch.autograd.Variable(Y)    
        
        if self.is_autoencoder:
            Y = (Y, X)
        
        
        return X,Y
    
    
    def __iter__(self):
        
        snippet_half = self.snippet_size//2
        
        outputs = []
        self.pbar = tqdm.tqdm(self.egg_g)
        for i_bn, v_eggs in self.pbar:
            row = v_eggs.iloc[0]
            if self.local_dir:
                fdir = self.local_dir
            else:
                fdir = row['results_dir']
            
            
            bn = os.path.join(fdir, row['base_name'])
            feat_file = bn + '_skeletons.hdf5'
            mask_file = bn + '.hdf5'
            
            if not os.path.exists(mask_file) or not os.path.exists(feat_file):
                #print('File does not exists!!!', mask_file)
                continue
            
            
            with pd.HDFStore(feat_file, 'r') as fid:
                trajectories_data = fid['/trajectories_data']
            trajectories_data.index = trajectories_data['frame_number']
            first_frame = trajectories_data.index.min()
            last_frame = trajectories_data.index.max()
            assert first_frame == 0
            assert last_frame == trajectories_data.shape[0]-1
            
            
            pos_index = v_eggs['frame_number'].values
            pos_index = pos_index.tolist()
            
            i1 = first_frame + snippet_half
            i2 = last_frame - snippet_half
            
            neg_index = []
            while len(neg_index) < len(pos_index):
                ind = random.randint(i1, i2)
                if not ind in pos_index:
                    neg_index.append(ind)
            
            
            if self.select_near_event:
                # I do not want indexes in the current index
                bad_ind = sum([list(range(pp-1, pp+2)) for pp in pos_index], []) 
                
                for pp in pos_index:
                    p1 = pp - 10
                    p2 = pp + 10
                    
                    ind = random.randint(p1, p2)
                    
                    test_n = 0
                    while ind in bad_ind:
                        # this condtions could create an infinite loop if not checked
                        ind = random.randint(p1, p2)
                        test_n += 1
                        if test_n > 10:
                            break
                    neg_index.append(ind)
            
            
            labels = len(pos_index)*[1] + len(neg_index)*[0]
            indexes = pos_index + neg_index
            
            
            with tables.File(mask_file) as fid:
                masks = fid.get_node('/mask')
                for lab, c_frame in  zip(labels, indexes):
                    
                    i1 = c_frame - snippet_half
                    i2 = c_frame + snippet_half + 1
                    snippet = masks[i1:i2, :, :]
                    
                    snippet = _shift_bgnd(snippet)
                    snippet = snippet.astype(np.float32)/255 - 0.5
                    
                    c_rows = trajectories_data[i1:i2]
                    if len(c_rows) != self.snippet_size:
                        continue
                    
                    
                    xx = c_rows['coord_x'].round().values
                    yy = c_rows['coord_y'].round().values
                    
                    roi_size = c_rows.iloc[0]['roi_size']
                    
                    snippet = _crop_and_augment(snippet, 
                                                xx, 
                                                yy, 
                                                roi_size, 
                                                self.roi_output_s, 
                                                is_nozoom = self.is_nozoom,
                                                is_bgnd_rm = self.is_bgnd_rm,
                                                is_augment = self.is_augment)
                    outputs.append((snippet, lab))
                    if len(outputs) >= self.n_batch:
                        yield self._prepare_torch(outputs[:self.n_batch])
                        outputs = outputs[self.n_batch:]
        
        if outputs:
            yield self._prepare_torch(outputs)
        
        return self

    
def _test():
    gen = EggLayingFlow(set_type = 'train')
    
    for X,Y in gen:
        print(X.size(), Y.size())

if __name__ == '__main__':
    #_test()
    import cProfile as profiler
    profiler.run('_test()', sort='tottime')
        
        