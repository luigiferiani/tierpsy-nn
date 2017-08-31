#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 19:14:13 2017

@author: ajaver
"""
import os
import cv2

from keras.models import load_model
from tierpsy.analysis.feat_food.getFoodContourNN import get_unet_prediction

main_dir = '/Users/ajaver/OneDrive - Imperial College London/egg_counter/example/'

model_t = load_model('unet_Adam-6-04449-0.2289.h5')

fnames = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.bmp')]
fname = fnames[0]

Xi = cv2.imread(fname, 0)

X_pred = get_unet_prediction(Xi, 
                  model_t, 
                  n_flips = 1,
                  im_size=None,
                  n_conv_layers = 3,
                  d4a_size = 24,
                  _is_debug=False)

#%%
import matplotlib.pylab as plt
import numpy as np

xx,yy = np.where(X_pred==1)

plt.imshow(Xi, cmap='gray')
plt.plot(yy,xx, '.')
