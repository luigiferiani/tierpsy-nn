#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""
import tqdm
import torch
from torch import nn
from flow import ROIFlowBatch
from model import MutualCNN, MutualLoss

if __name__ == '__main__':
    mask_file = '/Users/ajaver/OneDrive - Imperial College London/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5')
    
    n_epochs = 1
    
    model = MutualCNN()
    #criterion = MutualLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    gen = ROIFlowBatch(mask_file, feat_file)
    model.train()
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(gen)
        for input_var, target in pbar:
            output = model.forward(input_var)
            
            loss = criterion(output, target)
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
            
            dd = 'Epoch {} , loss={}'.format(epoch, loss.data[0])
            pbar.set_description(desc=dd, refresh=False)