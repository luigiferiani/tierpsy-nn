#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:05:20 2017

@author: ajaver
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class EggLConv3D(nn.Module):
    def __init__(self, 
                 snippet_size = 5,
                 embedding_size = 1):
        
        super().__init__()
        
        self.embedding_size = embedding_size
        self.snippet_size = snippet_size
        
        ini_kernel = min(snippet_size, 7)
        
        self.cnn = nn.Sequential(
            nn.Conv3d(snippet_size, 32, ini_kernel, padding=3),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            
            nn.Conv3d(32, 64, 3, stride=2, padding=(2,0,0)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            
            nn.MaxPool3d((1,2,2)),
            
            nn.Conv3d(64, 128, 3, stride=2, padding=(2,0,0)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            
            nn.MaxPool3d((1,2,2)),
            
            nn.Conv3d(128, 256, 3, stride=2, padding=(2,0,0)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            
            nn.MaxPool3d((2,2,2)),
        )
            
        self.fc = nn.Sequential( 
                nn.Linear(256, self.embedding_size),
                nn.Sigmoid()
                )
        
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def forward(self, x):
         x = self.cnn(x).view(-1, 256)
         x = self.fc(x)
         return x.view(-1)
#%%
class EggL_ChT(nn.Module):
    def __init__(self, snippet_size = 5,  embedding_size = 256):
        super().__init__()
        self.embedding_size = embedding_size
        self.snippet_size = snippet_size
        
        self.cnn = nn.Sequential(
            nn.Conv2d(self.snippet_size, 32, 7),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.MaxPool2d(2), 
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.MaxPool2d(2), 
            
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            
            nn.MaxPool2d(2), #b, 256, 1, 1
            
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(), 
        )
        self.fc =  nn.Sequential(
                nn.Linear(512, 128),
                nn.LeakyReLU(),
                nn.Dropout(p=0.15),
                nn.Linear(128, 32),
                nn.LeakyReLU(),
                nn.Dropout(p=0.15),
                nn.Linear(32, 1),
                nn.Dropout(p=0.15),
                nn.Sigmoid()
                )
        
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def forward(self, x):
        tt = x.size()
        x = x.view(*tt[:2], *tt[3:])
        x = self.cnn(x).view(-1, 512)
        x = self.fc(x)
        return x

#%%
class EggL_AE(nn.Module):
    def __init__(self, snippet_size = 5,  embedding_size = 256):
        super().__init__()
        self.embedding_size = embedding_size
        self.snippet_size = snippet_size
        self.cnn_encoder = nn.Sequential(
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
        self.fc_encoder = nn.Linear(256, self.embedding_size)
        self.fc_decoder = nn.Linear(self.embedding_size, 256)
        
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5),  # b, 256, 7, 7
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),  # b, 16, 15, 15
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),  # b, 16, 31, 31
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2),  # b, 8, 63, 63
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1),  # b, 1, 128, 128
            nn.Tanh()
        )
        
        self.lstm = nn.LSTM(input_size = embedding_size, 
                            hidden_size = 32, 
                            num_layers = 2, 
                            batch_first = True
                            )
        self.fc_class = nn.Sequential( 
                nn.Linear(32, 1),
                nn.Sigmoid()
                )
        
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def encoder(self, x):
         x = self.cnn_encoder(x).view(-1, 256)
         x = self.fc_encoder(x)
         return x
        
    def decoder(self, x):
        x = self.fc_decoder(x)
        x = x.view(-1, 256, 1, 1)
        x = self.cnn_decoder(x)
        return x
        
    def forward(self, x):
        
        x = x.view(-1, 1, 96, 96)
        embeddings = self.encoder(x)
        decoded = self.decoder(embeddings)
        decoded = decoded.view(-1, self.snippet_size, 1, 96, 96)
        
        emb_t = embeddings.view(-1, self.snippet_size,  self.embedding_size)
        out, _ = self.lstm(emb_t)
        lab = self.fc_class(out[:, -1, :]).view(-1)
        
        return lab, decoded
#%%
class FullLoss(nn.Module):
    def __init__(self, decode_loss_mix = 10., class_loss_mix = 0.1):
        super().__init__()

        self.decode_loss_mix = decode_loss_mix
        self.class_loss_mix = class_loss_mix
        
        self.class_loss_f = nn.BCELoss()
        self.decode_loss_f =  F.mse_loss
        
    def forward(self, outputs, targets):
        class_out, decoded = outputs
        class_target, input_v = targets
        
        self.class_loss = self.class_loss_f(class_out, class_target)
        self.decode_loss = self.decode_loss_f(decoded, input_v)
        
        d1 = self.decode_loss_mix*self.decode_loss
        d2 = self.class_loss_mix*self.class_loss
        
        loss = d1 + d2
        return loss
#%%
class STNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.Linear(10*8*8, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
       
    # Spatial transformer network forward function
    def forward(self, x):
        
        xs = self.cnn_loc(x)
        xs = xs.view(-1, 10*8*8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size())
        xo = F.grid_sample(x, grid)
        
        return xo
    
#%%
class EggL_STN(nn.Module):
    def __init__(self, snippet_size = 5,  embedding_size = 256):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.snippet_size = snippet_size
        
        self.cnn_encoder = nn.Sequential(
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
        self.fc_encoder = nn.Linear(256, self.embedding_size)
        
        self.stn = STNetwork()
        
        self.lstm = nn.LSTM(input_size = embedding_size, 
                            hidden_size = 32, 
                            num_layers = 2, 
                            batch_first = True
                            )
        self.fc_class = nn.Sequential( 
                nn.Linear(32, 1),
                nn.Sigmoid()
                )
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        
    def forward(self, x):
        #flatten snippets
        x = x.view(-1, 1, 96, 96)
        
        #i hope the stn center the data before doing the time difference
        xc = self.stn(x)
        
        embeddings = self.cnn_encoder(xc).view(-1, 256)
        embeddings = self.fc_encoder(embeddings)
        
        emb_t = embeddings.view(-1, self.snippet_size,  self.embedding_size)
        out, _ = self.lstm(emb_t)
        lab = self.fc_class(out[:, -1, :]).view(-1)
        
        return lab
        
class EggL_STN_oflow(EggL_STN):
    def __init__(self, snippet_size = 5,  embedding_size = 256):
        super().__init__(snippet_size = snippet_size,  embedding_size = embedding_size)
        
        
    def forward(self, x):
        #flatten snippets
        x = x.view(-1, 1, 96, 96)
        
        #i hope the stn center the data before doing the time difference
        xc = self.stn(x)
        
        xc = xc.view(-1, self.snippet_size, 1, 96, 96)
        xc = xc[:, 1:] - xc[:, :-1]
        xc = xc.view(-1, 1, 96, 96)
        
        embeddings = self.cnn_encoder(xc).view(-1, 256)
        embeddings = self.fc_encoder(embeddings)
        
        emb_t = embeddings.view(-1, self.snippet_size-1,  self.embedding_size)
        out, _ = self.lstm(emb_t)
        lab = self.fc_class(out[:, -1, :]).view(-1)
        
        return lab
#%%
class EggL_Diff(nn.Module):
    def __init__(self, snippet_size = 5,  embedding_size = 256):
        super().__init__()
        self.embedding_size = embedding_size
        self.snippet_size = snippet_size
        self.cnn_encoder = nn.Sequential(
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
        self.fc_encoder = nn.Linear(256, self.embedding_size)
        
        self.lstm = nn.LSTM(input_size = embedding_size, 
                            hidden_size = 32, 
                            num_layers = 2, 
                            batch_first = True
                            )
        self.fc_class = nn.Sequential( 
                nn.Linear(32, 1),
                nn.Sigmoid()
                )
        
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def forward(self, x):
        
        x = x[:, 1:] - x[:, :-1]
        
        x = x.view(-1, 1, 96, 96)
        x = self.cnn_encoder(x).view(-1, 256)
        embeddings = self.fc_encoder(x)
        
        emb_t = embeddings.view(-1, self.snippet_size-1,  self.embedding_size)
        out, _ = self.lstm(emb_t)
        lab = self.fc_class(out[:, -1, :]).view(-1)
        
        return lab
#%% Two sources
class EggL_Diff_T2(nn.Module):
    def __init__(self, snippet_size = 5,  
                 embedding_size = 256):
        super().__init__()
        self.embedding_size = embedding_size
        self.snippet_size = snippet_size
        self.c_ind = snippet_size//2
        
        self.cnn_encoder_flow = nn.Sequential(
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
        
        self.cnn_encoder_img = nn.Sequential(
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
        
        self.fc_encoder = nn.Linear(256, self.embedding_size)
        
        self.lstm = nn.LSTM(input_size = embedding_size, 
                            hidden_size = 32, 
                            num_layers = 2, 
                            batch_first = True
                            )
        self.fc_class = nn.Sequential( 
                nn.Linear(32, 1),
                nn.Sigmoid()
                )
        
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def forward(self, x_in):
        
        xd = x_in[:, 1:] - x_in[:, :-1]
        xd = xd.view(-1, 1, 96, 96)
        
        xc = x_in[:, self.c_ind]
        
        x1 = self.cnn_encoder_flow(xd).view(-1, self.snippet_size-1, 256)
        x2 = self.cnn_encoder_img(xc).view(-1, 1, 256)
        #combine both streams
        x = x1+x2
        emb_t = self.fc_encoder(x).view(-1, 256)
        emb_t = emb_t.view(-1, self.snippet_size-1,  self.embedding_size)
        
        out, _ = self.lstm(emb_t)
        lab = self.fc_class(out[:, -1, :]).view(-1)
        
        return lab

#%%
if __name__ == '__main__':
    import tqdm
    from flow import EggLayingFlow
    
    is_cuda = True
    snippet_size = 5
    n_batch = 64
    embedding_size = 256
    
    #%%
#    mod = EggLConv3D(embedding_size = 1, 
#               snippet_size = snippet_size, 
#               n_batch = n_batch)
    
#    model = EggL_AE(embedding_size = embedding_size)
#    criterion = FullLoss()
#    model = EggL_STN(embedding_size = embedding_size)
#    model = EggL_STN(embedding_size = embedding_size)

    model = EggL_Diff_T2(embedding_size = embedding_size)
    criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if is_cuda:
        model.cuda()
        criterion.cuda()
    
    gen = EggLayingFlow(set_type = 'test',
                   snippet_size = snippet_size, 
                   n_batch = n_batch,
                   is_cuda = is_cuda,
                   is_augment = False,
                   is_bgnd_rm = True,
                   is_autoencoder = False,
                   select_near_event = False)
    
    for ii in range(100):  
        pbar = tqdm.tqdm(gen)
        for in_v, target in gen:
            
            out = model(in_v)
            loss = criterion(out, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            pred1 = ((target[0]>0) == (out[0]>0.5)).float().mean().mean()
            
            ss = 'epoch {} : loss {}, pred1 {:.2f}'.format(ii, loss.data[0], pred1.data[0])
            gen.pbar.set_description(ss, refresh=False)
        
    #%%
#    import matplotlib.pylab as plt
#    import numpy as np
#    
#    xc = model.std(in_v.view(-1, 1, 96, 96)).view(-1, model.snippet_size, 1, 96, 96)
#    
#    xreal = in_v.data.cpu().squeeze().numpy()
#    xrealc = xc.data.cpu().squeeze().numpy()
#    
#    
#    for mm in range(xreal.shape[0]):
#        plt.figure(figsize=(18, 10))
#        for tt in range(xreal.shape[1]):
#            plt.subplot(2,5, tt+1)
#            plt.imshow(xreal[mm, tt])
#            plt.subplot(2,5, 5 + tt+1)
#            plt.imshow(xrealc[mm, tt])
#        plt.suptitle('R{}, P{}'.format(target.data[mm], out.data[mm]))
    
    
    