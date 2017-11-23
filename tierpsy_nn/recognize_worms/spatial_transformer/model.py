#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:10:23 2017

@author: ajaver
"""
import torch
from torch import nn
import torch.nn.functional as F


def weights_init_xavier(m):
    '''
    Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.uniform(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)

class CNNClf(nn.Module):
    def __init__(self, num_output):
        super().__init__()
        self.cnn_clf = nn.Sequential(
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
        # Regressor to the classification labels
        self.fc_clf = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, num_output)
        )
        
        for m in self.modules():
            weights_init_xavier(m)
        
        
    def forward(self, x):
        # transform the input
        x = self.cnn_clf(x).view(-1, 256)
        x = self.fc_clf(x)
        
        return x        

class STNetwork(CNNClf):
    def __init__(self, num_output):
        super().__init__(num_output)
        self.cnn_loc = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10*16*16, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)
        )
        
        for m in self.modules():
            weights_init_xavier(m)
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.fill_(0)
        self.fc_loc[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
           
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.cnn_loc(x)
        xs = xs.view(-1, 10*16*16)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size())
        xo = F.grid_sample(x, grid)
        
        return xo

    def forward(self, x):
        # transform the input
        xs = self.stn(x)
        xs = self.cnn_clf(xs).view(-1, 256)
        xs = self.fc_clf(xs)
        
        return xs




if __name__ == '__main__':
    from flow import LabelFlow

    import tqdm
    import numpy as np
    
    n_classes = 6
    net = STNetwork(n_classes)
    
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    gen = LabelFlow()
    gen.tiny()
    net.train()
    
    n_epochs = 1000
    for epoch in range(n_epochs):
        
        pbar = tqdm.tqdm(batchify(gen))
        for input_var, target_var in pbar:
            #output = net.forward(input_var)
            #loss = criterion(output, target_var-1) 
            
            X = input_var.data.squeeze().numpy()
            X = np.concatenate([random_rotation(x)[None, None, ...] for x in X])
            X = torch.autograd.Variable(torch.from_numpy(X).float())
            
            output = net.stn(X)
            
            loss = criterion(output, input_var) 
            
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()       
            
            #acc = ((output.max(1)[1]+1) == target_var).float().mean()
            #dd = 'Epoch {} , loss={}, acc={}'.format(epoch, loss.data[0], acc.data[0])
            dd = 'Epoch {} , loss={}'.format(epoch, loss.data[0])
            pbar.set_description(desc=dd, refresh=False)

    #%%
    net.test()
    imgs = input_var.data.squeeze().numpy()
    imgs_r = X.data.squeeze().numpy()
    imgs_t = output.data.squeeze().numpy()
    import matplotlib.pylab as plt
    for ii in range(imgs.shape[0]):
        plt.figure(figsize=(12,5))
        plt.subplot(1,3,1)
        plt.imshow(imgs[ii], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.title('original')
        plt.subplot(1,3,2)
        plt.imshow(imgs_r[ii], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.title('rotated')
        plt.subplot(1,3,3)
        plt.imshow(imgs_t[ii], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.title('stn')
        break    
    
    