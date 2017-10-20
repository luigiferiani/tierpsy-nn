#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:15:26 2017

@author: ajaver
"""
import tables
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pylab as plt
import itertools

from sklearn.metrics import confusion_matrix
from collections import Counter

from skeletons_flow import SkeletonsFlow, _h_angles
from train_model import _h_get_paths, reduced_strains, wild_isolates_old, CeNDR_base_strains, wild_isolates_WT2

#%%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    cmap=plt.cm.Blues
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    np.set_printoptions(precision=2)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%1.2f' % cm[i, j],
                 horizontalalignment="center",
                 fontsize =12,
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#%%
if __name__ == '__main__':
    
#    model_path = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs_sever/logs/W_resnet50_D0.0_20170928_112527/W_resnet50_D0.0-0233-1.6491.h5'
#    valid_strains = wild_isolates_old
#    sample_size_frames_s = 90
#    sample_frequency_s = 1/10.
#    is_angle = False
    
#    model_path = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs_sever/logs/resnet50_20170925_225727/resnet50-0077-4.1958.h5'
#    valid_strains = None
#    sample_size_frames_s = 90
#    sample_frequency_s = 1/10.
#    is_angle = False
    
#    model_path = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs_sever/logs/W_ang_resnet50_D0.0_20171005_120834/W_ang_resnet50_D0.0-0311-1.6940.h5'
#    valid_strains = wild_isolates_old
#    sample_size_frames_s = 90
#    sample_frequency_s = 1/10.
#    is_angle = True
    
#    model_path = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/CeNDR/logsN/S90_F0.1_R_ang_resnet50_D0.0_20171018_212909/S90_F0.1_R_ang_resnet50_D0.0-0259-1.8894.h5'
#    valid_strains = CeNDR_base_strains
#    is_angle = True
#    is_CeNDR = True

    model_path = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/CeNDR/logsN/S90_F0.1_ang_resnet50_D0.0_20171018_191655/S90_F0.1_ang_resnet50_D0.0-0099-2.8045.h5'
    valid_strains = None
    is_angle = True
    is_CeNDR = True

    if not is_CeNDR:
        base_file = 'SWDB_skel_smoothed.hdf5'
    else:
        base_file = 'CeNDR_skel_smoothed.hdf5'
    log_dir_root, main_file = _h_get_paths(base_file)
    
    
    print('loading model...')
    model = load_model(model_path)
     
    print('loading data...')
    
    gen = SkeletonsFlow(main_file = main_file, 
                       n_batch = 32, 
                       set_type='test',
                       valid_strains = valid_strains,
                       is_angle = is_angle
                       )
    expected_fps = 30
    #%%
    all_results = []
    for ii, (irow, row) in enumerate(gen.skeletons_indexes.iterrows()):
        print(ii+1, len(gen.skeletons_indexes))
        
        if 'fps' in row:
            fps = row['fps']
        else:
            fps = expected_fps
        
        
        #get the expected row indexes
        row_indices_r = np.linspace(0, gen.sample_size_frames_s, gen.n_samples)*fps
        
        
        row_skels = []
        
        sample_size_frames = int(gen.sample_size_frames_s*fps)
        for ini_r in range(row['ini'], row['fin'], int(sample_size_frames/2)):
            fin_r = ini_r + sample_size_frames
            if fin_r > row['fin']:
                continue 
            
            row_indices = row_indices_r + ini_r
            row_indices = np.round(row_indices).astype(np.int32)
            
            with tables.File(gen.main_file, 'r') as fid:
                skeletons = fid.get_node('/skeletons_data')[row_indices, :, :]
            body_coords = np.mean(skeletons[:, gen.body_range[0]:gen.body_range[1]+1, :], axis=1)
            
            if not is_angle:
                skeletons -= body_coords[:, None, :]
                row_skels.append(skeletons)
            else:
                X, _ = _h_angles(skeletons)
                X = X[..., None]
                row_skels.append(X)
                
        batch_data = np.array(row_skels)
        Y = model.predict(batch_data)
        
        all_results.append((row, Y))
    #%%
    with pd.HDFStore(gen.main_file, 'r') as fid:
        strains_codes = fid['/strains_codes']
    #%%
    y_vec_dict = {}
    for row, y_l in all_results:
        strain_id = row['strain']
        
        y = np.sum(y_l, axis=0)
        if not strain_id in y_vec_dict:
            y_vec_dict[strain_id]  = y
        else:
             y_vec_dict[strain_id] += y
    
    #%%
    for strain, vec in y_vec_dict.items():
        prob = vec/np.sum(vec)
        ind = np.argsort(prob)[:-6:-1]
        s_sort = strains_codes.loc[ind, 'strain']
        
        print(strain)
        for s, p in zip(s_sort, prob[ind]):
            print('{} - {:.2f}'.format(s,p*100))
        print('*************')
        
        
    #%%
    
    
    
    y_pred_dict = {}
    for row, y_l in all_results:
        yys = np.argmax(y_l, axis=1)
        
        exp_id = row['experiment_id']
        if not exp_id in y_pred_dict:
            y_pred_dict[exp_id]  = []
        
        y_pred_dict[exp_id] += list(yys)
    
    #%%
    df, _ = zip(*all_results)
    df = pd.concat(df, axis=1).T
    
    y_true, y_pred = [], []
    
    chuck_p = []
    for _, row in df.drop_duplicates('experiment_id').iterrows():
        y_true.append(row['strain'])
        
        y_l = y_pred_dict[row['experiment_id']]
        
        dd = Counter(y_l).most_common(1)[0][0]
        y_pred.append(strains_codes.loc[dd]['strain']) 
        
        chuck_p += [(row['strain'], x) for x in strains_codes.loc[y_l]['strain'].values]
    #%%
    labels = sorted(list(set(y_true)))
    dd = sum(x[0] == x[1] for x in chuck_p)
    print('Accuracy by chunk: {}'.format(dd/len(chuck_p)))
    #%%
    
    
    cm_c = confusion_matrix(*zip(*chuck_p), labels=labels)
    plt.figure(figsize=(21,21))
    plot_confusion_matrix(cm_c, 
                          labels,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          normalize = True
                          )
    
    #%%
    dd = sum(x[0] == x[1] for x in zip(y_pred, y_true))
    print('Accuracy by video: {}'.format(dd/len(y_true)))
    #%%
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(21,21))
    plot_confusion_matrix(cm, 
                          labels,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          normalize = True
                          )
    
    #%%
#    for rr, pp in zip(y_real, y_pred_dict):
#        dd = sorted(pp.items(), key = lambda x : x[1])[::-1]
#        
#        print('*** {} ***'.format(rr))
#        for k,v in dd:
#            print('{} : {:.1f}'.format(k, v*100))
        