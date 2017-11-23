#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:10:23 2017

@author: ajaver
"""
import os
import shutil
import torch
import warnings
import tensorflow as tf #for tensorboard
from sklearn.metrics import f1_score
import numpy as np
from collections import OrderedDict

def metrics_avg(m):
    dd = [list(zip(*x)) for x in zip(*m)]
    return OrderedDict([(x[0][0], np.mean(x[1])) for x in dd])


def pbar_description(metrics, epoch, is_train):
    m_str = ', '.join(['{}: {:.3f}'.format(*x) for x in metrics])
    d_str = 'Epoch : {} | {}'.format(epoch, m_str)
    if not is_train:
        'Val ' + d_str
        
    return d_str

def get_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, tuple):
        output = output[0]
    
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    #calculate the global f1 score
    #prefer to use scikit instead of having to program it again in torch
    ytrue = target.data.cpu().numpy()
    ypred = pred.data[0].cpu().numpy()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = f1_score(ytrue, ypred, average='macro')
    return res, f1

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)

class TBLogger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        
