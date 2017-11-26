#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:10:23 2017

@author: ajaver
"""
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import sys
import tqdm
import os
import time

from flow import LabelFlow
from model import STNetwork, CNNClf
import train_helper as th

def _get_log_dir(model_name, details=''):
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/recognize_worms/results'
    else:
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/recognize_worms/logs/'
    
    _BASEDIR = os.path.dirname(os.path.abspath(__file__))
    pdir = os.path.split(_BASEDIR)[-1]
    log_dir_root = os.path.join(log_dir_root, pdir)

    log_dir = os.path.join(log_dir_root, '{}_{}_{}'.format(model_name, details, time.strftime('%Y%m%d_%H%M%S')))
    return log_dir

def _metrics(output, target_var, loss, is_train=False):
        (prec1,), f1 = th.get_accuracy(output, target_var, topk = (1,))
           
        tb = [('loss' , loss.data[0]),
            ('pred1' , prec1.data[0]),
            ('f1' , f1)
            ]
        prefix = 'train_' if is_train else 'test_'
        tb = [(prefix + x, y) for x,y in tb]
        
        return tb
    
def train_epoch(epoch, is_cuda, model, train_loader, criterion, optimizer):
    model.train()
    pbar = tqdm.tqdm(train_loader)
    all_metrics = []
    for input_var, target_var in pbar:
        input_var, target_var = Variable(input_var), Variable(target_var.squeeze())
        if is_cuda:
            input_var =  input_var.cuda()
            target_var = target_var.cuda()
        
        output =  model(input_var)
        loss =  criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        m = _metrics(output, target_var, loss, is_train=True)
        all_metrics.append(m)
        pbar.set_description(th.pbar_description(m, epoch, is_train=False), refresh=False)
        
    return th.metrics_avg(all_metrics)

def test_epoch(epoch, is_cuda, model, test_loader, criterion, optimizer):
    model.eval()
    pbar = tqdm.tqdm(test_loader)
    
    all_metrics = []
    for input_var, target_var in pbar:
        #I do not need gradients here
        input_var, target_var = Variable(input_var, volatile=True), Variable(target_var.squeeze())
        if is_cuda:
            input_var =  input_var.cuda()
            target_var = target_var.cuda()
            
        output = model(input_var)
        loss = criterion(output, target_var)
        
        m = _metrics(output, target_var, loss, is_train=False)
        all_metrics.append(m)
        pbar.set_description(th.pbar_description(m, epoch, is_train=False))
    return th.metrics_avg(all_metrics)

def fit(model,
         optimizer,
         criterion,
         train_loader,
         test_loader,
         n_epochs,
         log_dir,
         is_cuda):
    
    logger = th.TBLogger(log_dir)
    best_f1 = 0
    for epoch in range(1, n_epochs + 1):
        train_metrics =  train_epoch(epoch, is_cuda, model, train_loader, criterion, optimizer)
        #write metrics of the training epochs
        for tag, value in train_metrics.items():
            logger.scalar_summary(tag, value, epoch)
        
        
        test_metrics = test_epoch(epoch, is_cuda, model, test_loader, criterion, optimizer)
        
        val_f1 = test_metrics['test_f1']
        is_best = val_f1 > best_f1
        best_f1 = max(val_f1, best_f1)
        
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_f1': val_f1,
            'optimizer' : optimizer.state_dict(),
        }
        th.save_checkpoint(state, is_best, save_dir = log_dir)
        for tag, value in test_metrics.items():
            logger.scalar_summary(tag, value, epoch)

    
def main(model_name,
        batch_size = 64,
        n_epochs = 1000
        ):
    is_cuda = torch.cuda.is_available()
    
    if sys.platform == 'linux':
        data_dir = os.environ['TMPDIR']
    else:        
        data_dir = '/Users/ajaver/OneDrive - Imperial College London/recognize_worms/'
    samples_file = os.path.join(data_dir, 'worm_ROI_samplesI.hdf5')
    
    #flag to check if cuda is available
    is_cuda = torch.cuda.is_available()
    log_dir = _get_log_dir(model_name)
    
    train_dataset = LabelFlow(samples_file, 'train', is_shuffle=True, is_cuda=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    
    test_dataset = LabelFlow(samples_file, 'test', is_shuffle=False, is_cuda=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    
    if model_name == 'STN':
        model = STNetwork(train_dataset.num_classes)
    elif model_name == 'Simple':
        model = CNNClf(train_dataset.num_classes)
    else:
        raise ValueError
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if is_cuda:
        print('This is CUDA!!!!')
        model = model.cuda()
        criterion = criterion.cuda()
        
    fit(model,
         optimizer,
         criterion,
         train_loader,
         test_loader,
         n_epochs,
         log_dir,
         is_cuda)

if __name__ == '__main__':
    main()