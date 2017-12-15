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
    
    model = EggL_AE(embedding_size = embedding_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = FullLoss()
    if is_cuda:
        model.cuda()
        criterion.cuda()
    
    gen = EggLayingFlow(set_type = 'test',
                   snippet_size = snippet_size, 
                   n_batch = n_batch,
                   is_cuda = is_cuda,
                   is_autoencoder = True)
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
        
    
    