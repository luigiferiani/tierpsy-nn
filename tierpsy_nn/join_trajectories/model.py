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

class ContrastiveLoss(torch.nn.Module):
    """
    https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label.float()
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
    
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

class SiameseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.fc = nn.Linear(self.cnn.embedding_size, 2)
        
    def forward(self, input_var):
        x1,x2 = input_var
        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        return x1, x2
        
        