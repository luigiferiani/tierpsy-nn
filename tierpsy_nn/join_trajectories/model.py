#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:08:01 2017

@author: ajaver
"""
import torch
from torch import nn
from torch.nn import functional as F
import math

class MutualLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        x1, x2 = output
        l1 = torch.abs(x1-x2).mean(1)
        t = F.sigmoid(l1)
        return F.binary_cross_entropy(t, target)
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_size = 64
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2), #b, 256, 1, 1
        )
        
        self.fc = nn.Linear(256, self.embedding_size)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    
    def forward(self, x_in):
        x = self.encoder(x_in).view(-1, 256)
        x = self.fc(x)
        return x

class MutualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.fc = nn.Linear(self.cnn.embedding_size, 2)
        
    def forward(self, input_var):
        x1,x2 = input_var
        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        x = self.fc(x1-x2)
        return x
        
        