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
import numpy as np
from skimage.transform import rescale, rotate

import torch


from torch.utils.data import Dataset, DataLoader

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

class LabelFlow(Dataset):
    labels = {1:'BAD', 2:'WORM', 3:'DIFICULT_WORM', 4:'WORM_AGGREGATE', 5:'EGGS', 6:'LARVAE'}
    mode = ''
    transform_funcs = [random_zoom_out, random_rotation, random_h_flip, random_v_flip]

    def __init__(self, 
                 samples_file,
                 mode, 
                 is_shuffle = False,
                 is_cuda=False, 
                 frac_test=0.1):
        
        super().__init__()
        self.is_cuda = is_cuda
        self.is_shuffle = is_shuffle
        self.samples_file = samples_file
        
        with pd.HDFStore(self.samples_file, 'r') as fid:
            sample_data = fid['/sample_data']
            sample_data = sample_data[sample_data['label_id']>= 1] 
            
        self.masks_fid = tables.File(self.samples_file, 'r')
        self.masks = self.masks_fid.get_node('/mask')
        self.full_img = self.masks_fid.get_node('/full_data')
        
        self.sample_data = sample_data
        self.mode = mode


        self._split_indexes(frac_test)        
        
    def _split_indexes(self, frac_test):
        random.seed(777) #to be sure we are making always the same subdivisions
        group_by_label = self.sample_data.groupby('label_id')
        mode_index = {'train':{}, 'test':{}, 'tiny':{}}
        for g, ind in group_by_label.groups.items():
            tot = ind.size
            
            #the labels are already randomized so I don't care to do it now
            ii = int(math.ceil(tot*frac_test))
            mode_index['test'][g] = list(ind[:ii])
            mode_index['train'][g] = list(ind[ii:])
            mode_index['tiny'][g] = list(ind[ii:ii+6])
        
        self._tot = {}
        for k,dat in mode_index.items():
            self._tot[k] = sum(len(x) for x in dat.values())
            
        self._mode_index = mode_index
        
        #get all availabe conviations as indexes
        self._indexes = []
        for l_id in self._mode_index[self.mode]:
            for ind in self._mode_index[self.mode][l_id]:
                for is_mask in [True, False]:
                    self._indexes.append((l_id, ind, is_mask))
        
    
    def _get_roi(self, ind, is_mask):
        lab = self.sample_data.loc[ind, 'label_id']
        
        if is_mask:
            img = self.masks[ind]
        else:
            img = self.full_img[ind]
            
        return img, lab
    
    def _to_torch(self, X, Y):
        X = X[ None, ...].astype(np.float32)/255
        Y = np.array([Y])
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()
        
        
        return X,Y
    
    def _transform(self, img):
        for func in self.transform_funcs:
            img = func(img)
        return img
    
    def __getitem__(self, idx):
        if self.is_shuffle:
            labels_id = list(self._mode_index[self.mode].keys())
            l_id = random.choice(labels_id)
            ind = random.choice(self._mode_index[self.mode][l_id])
            is_mask = random.choice([True, False])
            img, lab = self._get_roi(ind, is_mask)
            img = self._transform(img)
        else:
            l_id, roi_ind, is_mask = self._indexes[idx]
            img, lab = self._get_roi(roi_ind, is_mask)
        
        #I need to substract 1 of lab because the label_ids start with 1
        lab -= 1
        return self._to_torch(img, lab)
    
    def __len__(self):
        #I add two to include the 
        return 2*self._tot[self.mode]
    
    @property
    def num_classes(self):
        return len(self.labels)
if __name__ == '__main__':
    roi_dataset = LabelFlow('tiny', is_shuffle=False, is_cuda=False)
    X,Y = roi_dataset[10]

    dataloader = DataLoader(roi_dataset, batch_size=32,
                        shuffle=True, num_workers=0)
    for i_batch, (X,Y) in enumerate(dataloader):
        print(i_batch, X.size(),Y.size())
    
    #X = Variable(X)
    #Y = Variable(Y)
    
#    def __iter__(self):
#        
#        if self.mode == 'train':
#            labels_id = list(self._set_index[self.mode].keys())
#            for _ in range(len(self)):
#                l_id = random.choice(labels_id)
#                ind = random.choice(self._set_index[self.mode][l_id])
#                is_masks = random.choice([True, False])
#                yield self._get_roi(ind, is_masks)
#        
#        
#        else:
#            for l_id in self._set_index[self.mode]:
#                for ind in self._set_index[self.mode][l_id]:
#                    for is_mask in [True, False]:
#                        yield self._get_roi(ind, is_mask)
        
        
    
    
        
    

        
    
    
#def batchify(gen, batch_size=32, is_torch=True):
#    def _make_batch(dat):
#        X,Y = zip(*dat)
#        Y = np.array(Y)
#        X = np.concatenate(X)
#        if is_torch:
#            X = X[:, None, ...].astype(np.float32)/255
#            X = torch.from_numpy(X)
#            Y = torch.from_numpy(Y)
#            if gen.is_gpu:
#                X = X.cpu()
#                Y = Y.cpu()
#                
#            X = Variable(X)
#            Y = Variable(Y)
#        return X, Y
#    
#    remainder = []
#    for x,y in gen:
#        remainder.append((x[None, ... ], y))
#        
#        if len(remainder) >= batch_size:
#            chunk = _make_batch(remainder[:batch_size])
#            remainder = remainder[batch_size:] 
#            yield chunk
#    if remainder:
#        yield _make_batch(remainder)
#    
#    
#    
##%%
#if __name__ == '__main__':
#    g = LabelFlows()
#    g.train()
#    for ii, (X,Y) in enumerate(batchify(g)):
#        print(X.size(), Y.size())
#        break
#        
#        
#        
#    
#    
#        