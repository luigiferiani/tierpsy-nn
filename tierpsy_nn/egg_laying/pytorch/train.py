#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:10:23 2017

@author: ajaver
"""
import torch
from torch import nn
import sys
import tqdm
import os
import time

from flow import EggLayingFlow
from model import EggL_AE, FullLoss #EggLConv3D
import train_helper as th

def _get_log_dir(model_name, details=''):
    if sys.platform == 'linux':
        log_dir_root = os.path.join(os.environ['HOME'], 'egg_laying', 'results')
    else:
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/egg_laying/logs/'
    
    _BASEDIR = os.path.dirname(os.path.abspath(__file__))
    pdir = os.path.split(_BASEDIR)[-1]
    log_dir_root = os.path.join(log_dir_root, pdir)

    log_dir = os.path.join(log_dir_root, '{}_{}_{}'.format(model_name, details, time.strftime('%Y%m%d_%H%M%S')))
    return log_dir

def _metrics(output, target_var, loss, is_train=False):
    if isinstance(target_var, (tuple, int)):
        target_var = target_var[0]
        output = output[0]
        
    
    pred1 = ((target_var>0) == (output>0.5)).float().mean()
    tb = [('loss' , loss.data[0]),
        ('pred1' , pred1.data[0])
        ]
    prefix = 'train_' if is_train else 'test_'
    tb = [(prefix + x, y) for x,y in tb]
    
    return tb

def train_epoch(epoch, is_cuda, model, train_loader, criterion, optimizer):
    model.train()
    all_metrics = []
    for input_var, target_var in train_loader:
        output =  model(input_var)
        loss =  criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        m = _metrics(output, target_var, loss, is_train=True)
        all_metrics.append(m)
        train_loader.pbar.set_description(th.pbar_description(m, epoch, is_train=True), refresh=False)
    
    avg_m = th.metrics_avg(all_metrics)
    train_loader.pbar.set_description(th.pbar_description(avg_m.items(), epoch, is_train=True), refresh=True)
    
    return avg_m

def test_epoch(epoch, is_cuda, model, test_loader, criterion, optimizer):
    model.eval()
    
    all_metrics = []
    for input_var, target_var in test_loader:
        output = model(input_var)
        loss = criterion(output, target_var)
        
        m = _metrics(output, target_var, loss, is_train=False)
        all_metrics.append(m)
        test_loader.pbar.set_description(th.pbar_description(m, epoch, is_train=False))
        
    avg_m = th.metrics_avg(all_metrics)
    test_loader.pbar.set_description(th.pbar_description(avg_m.items(), epoch, is_train=False), refresh=True)
    
    return avg_m
    
def fit(model,
         optimizer,
         criterion,
         train_loader,
         test_loader,
         n_epochs,
         log_dir,
         is_cuda):
    
    logger = th.TBLogger(log_dir)
    best_pred1 = 0
    for epoch in range(1, n_epochs + 1):
        train_metrics =  train_epoch(epoch, is_cuda, model, train_loader, criterion, optimizer)
        #write metrics of the training epochs
        for tag, value in train_metrics.items():
            logger.scalar_summary(tag, value, epoch)
        
        
        test_metrics = test_epoch(epoch, is_cuda, model, test_loader, criterion, optimizer)
        
        val_pred1 = test_metrics['test_pred1']
        is_best = val_pred1 > best_pred1
        best_pred1 = max(val_pred1, best_pred1)
        
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_pred1': val_pred1,
            'optimizer' : optimizer.state_dict(),
        }
        th.save_checkpoint(state, is_best, save_dir = log_dir)
        for tag, value in test_metrics.items():
            logger.scalar_summary(tag, value, epoch)

    
def main(model_name = 'EggSnippet',
         snippet_size = 5,
        roi_output_s = 96,
        n_batch = 64,
        n_epochs = 1000
        ):
    is_cuda = torch.cuda.is_available()
    
    if model_name == 'EggSnippet':
        model = EggL_AE(embedding_size = 256, 
                        snippet_size = snippet_size
                        )
        is_autoencoder = True
    else:
        raise ValueError
    
    model_name += '_nozoom'
    
    log_dir = _get_log_dir(model_name)
    
    train_flow = EggLayingFlow(set_type = 'train',
                                snippet_size = snippet_size,
                                roi_output_s = roi_output_s,
                                n_batch = n_batch,
                                is_augment = True,
                                is_cuda = is_cuda,
                                is_autoencoder = is_autoencoder)
    test_flow = EggLayingFlow(set_type = 'test',
                                snippet_size = snippet_size,
                                roi_output_s = roi_output_s,
                                n_batch = n_batch,
                                is_augment = False,
                                is_cuda = is_cuda,
                                is_autoencoder = is_autoencoder)
    
    
    
    criterion = FullLoss(decode_loss_mix = 10., class_loss_mix = 1.)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if is_cuda:
        print('This is CUDA!!!!')
        model = model.cuda()
        criterion = criterion.cuda()
        
    fit(model,
         optimizer,
         criterion,
         train_flow,
         test_flow,
         n_epochs,
         log_dir,
         is_cuda)

if __name__ == '__main__':
    import fire
    fire.Fire(main)