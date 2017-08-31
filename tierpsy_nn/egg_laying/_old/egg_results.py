#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:26:01 2017

@author: ajaver
"""
import os
import pickle
import glob
import numpy as np
import matplotlib.pylab as plt



def plot_probs(eggs, worm_probs_resized, worm_probs_fixed, thresh=0.99, plot_log=True):

    def get_possible_eggs(worm_probs, thresh ):
        inds = np.where(worm_probs[:,1]>thresh)[0]
        inds = inds[np.argsort(worm_probs[inds,1])][::-1]
        return inds
    
    def _plot_probs(worm_probs, thresh= 0.99, maker='v', col='b'):
        inds = get_possible_eggs(worm_probs, thresh)
        plt.plot(worm_probs[:,1], col)
        plt.plot(inds, worm_probs[inds,1], maker+col)
        
        return inds
    
    
    #eggs, worm_probs_resized, worm_probs_fixed = [results[x] for x in ['eggs', 'worm_probs_resized', 'worm_probs_fixed']]
    
    egg_frames = eggs['frame_number'].values
    
    #plt.figure()
    plt.subplot(2,1,1)
    _plot_probs(worm_probs_resized, thresh= thresh, maker='o', col='g')
    plt.plot(egg_frames, np.ones(len(egg_frames)), 'xr')
    plt.title('resized')
    
    plt.subplot(2,1,2)
    _plot_probs(worm_probs_fixed, thresh= thresh, maker='v', col='b')
    plt.plot(egg_frames, np.ones(len(egg_frames)), 'xr')
    plt.title('fixed')

if __name__ == '__main__':
    save_results_dir = './results'
    files_done = glob.glob(os.path.join(save_results_dir, '*_eggs.p'))

    for ii, fname in enumerate(files_done):
        print(ii, os.path.basename(fname))
        
        results = pickle.load(open(fname, "rb" ))
        plot_probs(**results)
        
        if ii > 10:
            break



    #%%
    egg_frames = results['eggs']['frame_number'].values

    inds_resized = np.where(results['worm_probs_fixed'][:,1]>0.99)[0]
    inds_fixed = np.where(results['worm_probs_resized'][:,1]>0.99)[0]
    #%%




