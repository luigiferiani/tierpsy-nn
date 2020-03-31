#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:10:43 2017

@author: ajaver
"""
import os

import torch
from torch.utils.data import DataLoader

from flow import LabelFlow
from model import STNetwork, CNNClf


if __name__ == '__main__':
    batch_size = 32

    data_dir = '/Users/ajaver/OneDrive - Imperial College London/recognize_worms/'
    samples_file = os.path.join(data_dir, 'worm_ROI_samplesI.hdf5')

    train_dataset = LabelFlow(samples_file, 'train', is_shuffle=True, is_cuda=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    test_dataset = LabelFlow(samples_file, 'test', is_shuffle=False, is_cuda=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    model_file = '/Users/ajaver/OneDrive - Imperial College London/recognize_worms/logs/spatial_transformer/STN__20171128_175315/checkpoint.pth.tar'
    model = STNetwork(train_dataset.num_classes)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    #%%
    for X,Y in train_loader:
        X = torch.autograd.Variable(X)
        Y = torch.autograd.Variable(Y)
        break

    Xt = model.stn(X)


    import matplotlib.pylab as plt
    x = X.data.squeeze().numpy()
    xt = Xt.data.squeeze().numpy()

    for nn in range(x.shape[0]):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(x[nn], interpolation='none', cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(xt[nn], interpolation='none', cmap='gray')