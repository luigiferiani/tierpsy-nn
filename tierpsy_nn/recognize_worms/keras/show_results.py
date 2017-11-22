#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:03:43 2017

@author: ajaver
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'gray'

#%%
def show_images(data, 
                labels = np.zeros([]),
                n_rows = 7, 
                lab_colors = ['red', 'green'], 
                lab_names = ['bad', 'worm']):

    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)
    
    plt.figure()
    tot_figs = min(n_rows*n_rows, data.shape[0])
    for ii in range(tot_figs): 
        ax = plt.subplot(n_rows,n_rows, ii+1, aspect='equal');
        img = np.squeeze(data[ii])
        img = ((img + 0.5)*255).astype(np.uint8)
        plt.imshow(img);
        plt.axis('off');
        
        if labels.size > 0:
            lab = labels[ii]
            ax.text(3, 8, lab_names[lab], 
                    bbox={'facecolor':lab_colors[lab], 'alpha':0.5, 'pad':1})
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

#%%
def show_bad(model, X, Y):
    pred = model.predict_proba(X)
    
    
    label_pred = np.argmax(pred, axis=1)
    label_real = np.argmax(Y, axis=1)    
    
    bad_labels = label_pred!=label_real
    
    
    bad_imgs = np.squeeze(X[bad_labels])
    if bad_imgs.ndim == 2:
        bad_imgs = bad_imgs[None, :, :]
    
    print('*****')
    print(sum(bad_labels), len(label_real))
    #%%
    show_images(bad_imgs, labels = label_pred[bad_labels])