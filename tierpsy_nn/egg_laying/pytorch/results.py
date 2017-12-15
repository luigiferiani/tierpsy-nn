#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:53:51 2017

@author: ajaver
"""
import torch

from model import EggL_AE
from flow import _crop_and_augment, EggLayingFlow

if __name__ == '__main__':
    import os
    
    
    model_w_dir = os.path.expanduser('~/egg_laying/results/pytorch/EggSnippet__20171214_202633/')
    
    model_file = os.path.join(model_w_dir, 'checkpoint.pth.tar')
    
    is_cuda = False
    
    
    mod = EggL_AE(embedding_size = 256, 
                        snippet_size = 5
                        )
    
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    mod.load_state_dict(checkpoint['state_dict'])
    mod.eval()
    
    if is_cuda:
        mod = mod.cuda()
    
    gen = EggLayingFlow(set_type = 'test',
                                snippet_size = 5,
                                roi_output_s = 96,
                                n_batch = 64,
                                is_augment = True,
                                is_cuda = is_cuda,
                                is_autoencoder = True)

    X,Y = next(iter(gen))
    
    
    
    
    
    lab, decoded = mod(X)
    #%%
    import numpy as np
    import matplotlib.pylab as plt
    
    xn = decoded.data.squeeze().numpy()
    
    xreal = X.data.squeeze().numpy()
    
    for nn in range(xn.shape[0]):
        plt.figure(figsize=(12, 5))
        for tt in range(xn.shape[1]):
            plt.subplot(2,5, tt+1)
            plt.imshow(xreal[nn, tt])
            
            plt.subplot(2,5, 5 + tt+1)
            plt.imshow(xn[nn, tt])
        
        plt.suptitle('R{}, P{}'.format(Y[0].data[nn], lab.data[nn]))
        
    
    
    