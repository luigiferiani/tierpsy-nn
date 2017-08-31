#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 20:45:50 2017

@author: ajaver
"""
import os
import glob
import numpy as np
from skimage.io import imread

main_dir = '/Users/ajaver/OneDrive - Imperial College London/food/train_set/'
fnames = glob.glob(os.path.join(main_dir, 'X_*'))


edges = np.arange(-255.5, 255.5)
bincounts = []
for ivid, fname in enumerate(fnames):
    print('{} of {}'.format(ivid+1, len(fnames)))
    
    img = imread(fname)
    dd = np.bincount(img.flatten(), minlength=256)
    
    dd_norm,_ = np.histogram(img-np.median(img), edges)
    
    bn = os.path.basename(fname)[:-4]
    bincounts.append((bn, dd, dd_norm))
    
#%%
import matplotlib.pylab as plt

counts = np.array([x[1] for x in bincounts])
counts_norm = np.array([x[2] for x in bincounts])
#%%
plt.figure()
plt.plot(counts.T)
#plt.plot(np.sum(counts, axis=0), 'k')
#%%
plt.figure()
#plt.plot(edges[1:]-0.5, np.sum(counts_norm, axis=0))
plt.plot(edges[1:]-0.5, counts_norm.T)