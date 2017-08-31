#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:11:22 2017

@author: ajaver
"""
import os
import numpy as np
import matplotlib.pylab as plt
import glob

save_dir = '/Users/ajaver/OneDrive - Imperial College London/egg_laying/results_N/'

fnames = glob.glob(os.path.join(save_dir, '*.npz'))

fname = fnames[-1]
data = np.load(fname)


Y_pred = data['arr_0']
Y_true = data['arr_1']

#%%
plt.figure()
for kk in range(Y_pred.shape[1]):
    plt.plot(Y_pred[:,kk,1], '.')
    plt.plot(Y_true, 'x')

