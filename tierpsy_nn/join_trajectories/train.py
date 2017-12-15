#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""
import os
import tqdm
import torch
from flow import ROIFlowBatch
from model import SiameseCNN, ContrastiveLoss

from torch.nn import functional as F

if __name__ == '__main__':
    #mask_file = '/Users/ajaver/OneDrive - Imperial College London/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/videos'
    fname = 'BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
    mask_file = os.path.join(data_dir,fname)
    feat_file = os.path.join(data_dir,fname.replace('.hdf5', '_featuresN.hdf5'))
    
    n_epochs = 1
    
    model = SiameseCNN()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    gen = ROIFlowBatch(mask_file, feat_file)
    model.train()
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(gen)
        for input_var, target in pbar:
            out1, out2 = model.forward(input_var)
            
            loss = criterion(out1, out2, target)
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
            
            pred = (F.pairwise_distance(out1, out2)> 1).long().squeeze()
            
            acc = (pred == target).float().mean()
            fones = (target.float().mean().data[0], pred.float().mean().data[0])
            
            dd = 'Epoch {} , loss={}, acc={}, frac_ones={:.2},{:.2}'.format(epoch, loss.data[0], acc.data[0], *fones)
            pbar.set_description(desc=dd, refresh=False)