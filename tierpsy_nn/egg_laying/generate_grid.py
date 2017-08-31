#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:08:16 2017

@author: ajaver
"""

import os

save_dir = './cmds'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



cmd = '''
#!/bin/sh
#PBS -l walltime=12:00:00
## This tells the batch manager to limit the walltime for the job to XX hours, YY minutes and ZZ seconds.

#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1
## This tells the batch manager to use NN node with MM cpus and PP gb of memory per node with QQ gpus available.

#PBS -q gpgpu
## This tells the batch manager to enqueue the job in the general gpgpu queue.

module load anaconda3
module load cuda
## This job requires CUDA support.

source activate tierpsy
KERAS_BACKEND=tensorflow python $HOME/nn_tests/egg_laying/train_eggs.py --window_size {} --y_offset_left {} --y_offset_right {} --is_tiny True
'''

window_sizes = [3, 5, 7]

for W in window_sizes:
    offset_max = W//2
    
    for offset in range(1, offset_max+1):
        
        ff = 'eggs_W{}-{}-{}.sh'.format(W, offset, offset)
        fname = os.path.join(save_dir, ff)
        
        with open(fname, 'w') as fid:
            fid.write(cmd.format(W, offset, offset))
