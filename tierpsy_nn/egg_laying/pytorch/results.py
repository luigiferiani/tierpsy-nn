#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:53:51 2017

@author: ajaver
"""

import os
import torch
import pandas as pd
import tables
import numpy as np
import matplotlib.pylab as plt
from model import EggL_Diff
from flow import _crop_and_augment, _shift_bgnd, EggLayingFlow, egg_events_file_dflt, local_dir_dlft

if __name__ == '__main__':
    
    
    is_cuda = True
    
    egg_events = pd.read_csv(egg_events_file_dflt)
    
    #model_w_dir = os.path.expanduser('~/egg_laying/results/pytorch/EggSnippet__20171214_202633/')
    #model_w_dir = os.path.expanduser('~/egg_laying/results/pytorch/_nozoom__20171215_162844/')
    #model_w_dir = os.path.expanduser('~/egg_laying/results/pytorch/EggL_Diff__20171218_143704')
    model_w_dir = os.path.expanduser('~/egg_laying/results/pytorch/EggL_Diff__20171218_210439')
    
    model_file = os.path.join(model_w_dir, 'checkpoint.pth.tar')
    mod = EggL_Diff(embedding_size = 256, 
                        snippet_size = 5
                        )
    
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    mod.load_state_dict(checkpoint['state_dict'])
    mod.eval()
    
    if is_cuda:
        mod = mod.cuda()
        
        
   #%%
#        
#    gen = EggLayingFlow(set_type = 'test',
#                                snippet_size = 5,
#                                roi_output_s = 96,
#                                n_batch = 8,
#                                is_augment = False,
#                                is_cuda = is_cuda,
#                                is_autoencoder = True)
#
#    X,Y = next(iter(gen))
#    
#    
#    lab, decoded = mod(X)
#    #%%
#    
#    xn = decoded.data.cpu().squeeze().numpy()
#    
#    xreal = X.data.cpu().squeeze().numpy()
#    
#    for nn in range(xn.shape[0]):
#        plt.figure(figsize=(15, 5))
#        for tt in range(xn.shape[1]):
#            plt.subplot(1,5, tt+1)
#            plt.imshow(xreal[nn, tt])
#    
#        plt.suptitle('R{}, P{}'.format(Y[0].data[nn], lab.data[nn])) 
    
    
    #%%
    df_files = egg_events.drop_duplicates(subset='base_name')[['set_type', 'results_dir', 'base_name']]
    #df_files = df_files[df_files['base_name']=='C11D2.2 (ok1565)IV on food R_2011_08_31__12_52_41___8___5']
    
    #row = df_files[df_files['set_type']=='test'].iloc[1]
    
    for irow, row in df_files[df_files['set_type']=='test'].iterrows():
        print(irow)
        #%%
        mask_file = os.path.join(local_dir_dlft, row['base_name'] + '.hdf5')
        feat_file = os.path.join(local_dir_dlft, row['base_name'] + '_skeletons.hdf5')
        
        with pd.HDFStore(feat_file, 'r') as fid:
            trajectories_data = fid['/trajectories_data']
        trajectories_data.index = trajectories_data['frame_number']
        first_frame = trajectories_data.index.min()
        last_frame = trajectories_data.index.max()
    
        #%%
        egg_v = egg_events[egg_events['base_name']==row['base_name']]['frame_number']
        
        egg_l_event = []
        labels = []
        n_batch = 50
        for tt in range(first_frame+2, last_frame-2, n_batch):
            #print(tt)
            with tables.File(mask_file, 'r') as fid:
                ini = tt-2
                fin = tt + n_batch + 2
                imgs = fid.get_node('/mask')[ini:fin]
                imgs = _shift_bgnd(imgs)
                
                imgs = imgs.astype(np.float32)/255 - 0.5
                
                c_rows = trajectories_data[ini:fin]
                xx = c_rows['coord_x'].round().values
                yy = c_rows['coord_y'].round().values
                roi_size = c_rows.iloc[0]['roi_size']
                
                imgs_c = _crop_and_augment(imgs, xx, yy, 
                          w_roi_size = roi_size, 
                          roi_output_size = 96, 
                          is_nozoom = True,
                          is_augment = False)
                img_c = np.hstack([imgs_c[ii : imgs_c.shape[0] - 5 + ii + 1] for ii in range(5)])
                img_c = img_c[:, :, None, :, :]
                
                img_c = torch.from_numpy(img_c).float()
                if is_cuda:
                    img_c = img_c.cuda()
                img_c = torch.autograd.Variable(img_c)
                lab = mod(img_c)
                
                if isinstance(lab, tuple):
                    lab = lab[0]
                
                frame_numbers = c_rows['frame_number'].values[2:-2]
                
                if any(x in frame_numbers for x in egg_v):
                    egg_l_event.append((frame_numbers, img_c.data.cpu().numpy(), lab.data.cpu().numpy()))
                    
                
                labels.append((frame_numbers, lab.data.cpu().numpy()))
                
                
        #%%
        egg_v = egg_events[egg_events['base_name']==row['base_name']]['frame_number']
        
        frame_numbers, ll = map(np.concatenate, zip(*labels))
        assert np.all(np.diff(frame_numbers)==1)
        
        plt.figure(figsize = (15, 5))
        plt.plot(frame_numbers, ll)
        plt.plot(egg_v, ll[egg_v-2], 'xr')
        break
        
        #%%
        
#        #%%
#        for ff, rois, labs in egg_l_event:
#            for e_ff in egg_v:
#                good = ff ==e_ff
#                roi  = rois[good].squeeze()
#                lab = labs[good].squeeze()
#                
#                if roi.size == 0:
#                    continue
#                
#                plt.figure(figsize=(15, 5))
#                for tt in range(5):
#                    plt.subplot(1, 5, tt+1)
#                    plt.imshow(roi[tt])
#                plt.suptitle('P{}'.format(lab)) 
#        #%%
#        
#        c_frame = 17900
#        with tables.File(mask_file, 'r') as fid:
#            ini = c_frame
#            fin = ini + n_batch + 2
#            imgs = fid.get_node('/mask')[ini:fin]
#        
#        mask = imgs>0
#        valid_pix = imgs[mask]
#        imgs[~mask] = np.percentile(valid_pix, 95)
#        
#        imgs = imgs.astype(np.float32)/255 - 0.5
#        
#        c_rows = trajectories_data[ini:fin]
#        xx = c_rows['coord_x'].round().values
#        yy = c_rows['coord_y'].round().values
#        roi_size = c_rows.iloc[0]['roi_size']
#        imgs_c = _crop_and_augment(imgs, xx, yy, 
#                          w_roi_size = roi_size, 
#                          roi_output_size = 96, 
#                          is_nozoom = True,
#                          is_augment = False)
#        img_c = np.hstack([imgs_c[ii : imgs_c.shape[0] - 5 + ii + 1] for ii in range(5)])
#        img_c = img_c[:, :, None, :, :]
#        
#        for tt, img_cs in enumerate(img_c):
#            tt_c = c_frame + tt + 2
#            roi_d = np.diff(img_cs.squeeze(), axis=0)
#            plt.figure(figsize=(15, 5))
#            for tt, roi_img in enumerate(roi_d):
#                plt.subplot(1, 5, tt+1)
#                plt.imshow(roi_img)
#            plt.suptitle(tt_c)

        
#%%
#    for nn in range(xn.shape[0]):
#        plt.figure(figsize=(18, 10))
#        dd = np.diff(xreal[nn], axis=0)
#        for tt in range(dd.shape[0]):
#            plt.subplot(2,5, tt+1)
#            plt.imshow(dd[tt])
#            plt.subplot(2,5, 5+ tt+1)
#            plt.imshow(xreal[nn, tt])
#        plt.suptitle('R{}, P{}'.format(Y[0].data[nn], lab.data[nn]))
                
    
            