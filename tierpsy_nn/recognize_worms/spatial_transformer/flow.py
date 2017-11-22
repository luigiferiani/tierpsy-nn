#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:10:23 2017

@author: ajaver
"""

import pandas as pd
import random
import math
import tables
import matplotlib.pylab as plt
import numpy as np
from skimage.transform import rescale, rotate

import torch
from torch.autograd import Variable

def random_zoom_out(img, zoom_out = 1.25):
    scale = random.uniform(1, zoom_out)
    
    img_r = rescale(img, scale, mode='symmetric')
    
    w, h = img_r.shape
    th, tw = img.shape
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    img_r = img_r[j:j+th, i:i+tw]
    assert img_r.shape == img.shape
    return img_r

def random_rotation(img, ang = 180):
    val = random.uniform(-ang, ang)
    img = rotate(img, val, resize=False, mode='symmetric')
    return img

def random_h_flip(img):
    if random.random() < 0.5:
        img = img[::-1, :]
    return img

def random_v_flip(img):
    if random.random() < 0.5:
        img = img[:, ::-1] 
    return img
 


class LabelFlows():
    samples_file = '/Users/ajaver/OneDrive - Imperial College London/training_data/worm_ROI_samplesI.hdf5'
    labels = {1:'BAD', 2:'WORM', 3:'DIFICULT_WORM', 4:'WORM_AGGREGATE', 5:'EGGS', 6:'LARVAE'}
    mode = ''
    transform_funcs = [random_zoom_out, random_rotation, random_h_flip, random_v_flip]
    
    def __init__(self, is_gpu=False, frac_test=0.9):
        self.is_gpu = is_gpu
        self.train()
        with pd.HDFStore(self.samples_file) as fid:
            sample_data = fid['/sample_data']
            sample_data = sample_data[sample_data['label_id']>= 1] 
            
        self.masks_fid = tables.File(self.samples_file, 'r')
        self.masks = self.masks_fid.get_node('/mask')
        self.full_img = self.masks_fid.get_node('/full_data')
        
        self.sample_data = sample_data
        self._split_indexes(frac_test)
        
    
    def _split_indexes(self, frac_test):
        group_by_label = self.sample_data.groupby('label_id')
        
        set_index = {'train':{}, 'test':{}}
        for g, ind in group_by_label.groups.items():
            tot = ind.size
            
            #the labels are already randomized so I don't care to do it now
            ii = int(math.ceil(tot*frac_test))
            set_index['test'][g] = list(ind[:ii])
            set_index['train'][g] = list(ind[ii:])
        
        
        self.tot_train = sum(len(v) for v in set_index['train'].values())
        self.test_train = sum(len(v) for v in set_index['test'].values())
        self._set_index = set_index
        
    
    def _get_roi(self, ind, is_mask):
        y = self.sample_data.loc[ind, 'label_id']
        if is_mask:
            img = self.masks[ind]
        else:
            img = self.full_img[ind]
            
        if self.mode == 'train':
            img = self._transform(img)
        
        return img, y
        
    
    
    def __iter__(self):
        
        if self.mode == 'train':
            labels_id = list(self._set_index['train'].keys())
            for _ in range(len(self)):
                l_id = random.choice(labels_id)
                ind = random.choice(self._set_index['train'][l_id])
                is_masks = random.choice([True, False])
                yield self._get_roi(ind, is_masks)
        
        
        elif self.mode == 'test':
            for l_id in self._set_index['test']:
                for ind in self._set_index['test'][l_id]:
                    for is_mask in [True, False]:
                        print(ind, is_mask)
                        yield self._get_roi(ind, is_mask)
        
    def test(self):
        self.mode = 'test'
        
    def train(self):
        self.mode = 'train'
        
        
    def __len__(self):
        #I add two to include the 
        if self.mode == 'train':
            return self.tot_train*2
        else:
            return self.tot_test*2
    
    def _transform(self, img):
        #random rotation
        for func in self.transform_funcs:
            img = func(img)
        return img
    
    
def batchify(gen, batch_size=32, is_torch=True):
    def _make_batch(dat):
        X,Y = zip(*dat)
        Y = np.array(Y)
        X = np.concatenate(X)
        if is_torch:
            X = X[:, None, ...].astype(np.float32)/255
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y)
            if gen.is_gpu:
                X = X.cpu()
                Y = Y.cpu()
                
            X = Variable(X)
            Y = Variable(Y)
        return X, Y
    
    remainder = []
    for x,y in gen:
        remainder.append((x[None, ... ], y))
        
        if len(remainder) >= batch_size:
            chunk = _make_batch(remainder[:batch_size])
            remainder = remainder[batch_size:] 
            yield chunk
    if remainder:
        yield _make_batch(remainder)
    
    
    
#%%
if __name__ == '__main__':
    g = LabelFlows()
    g.train()
    for ii, (X,Y) in enumerate(batchify(g)):
        print(X.size(), Y.size())
        break
        
        
        
    
    
        