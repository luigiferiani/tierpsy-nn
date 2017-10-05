#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:58:14 2017

@author: ajaver
"""
import os

cmd_str = '''
#!/bin/sh
#PBS -l walltime=00:30:00
## This tells the batch manager to limit the walltime for the job to XX hours, YY minutes and ZZ seconds.

#PBS -l select=1:ncpus=2:mem=8gb:ngpus=1
## This tells the batch manager to use NN node with MM cpus and PP gb of memory per node with QQ gpus available.

#PBS -q gpgpu
## This tells the batch manager to enqueue the job in the general gpgpu queue.

module load anaconda3
module load cuda
## This job requires CUDA support.
source activate tierpsy

## copy temporary files
cp $WORK/classify_strains/train_set/SWDB_skel_smoothed.hdf5 $TMPDIR/SWDB_skel_smoothed.hdf5

KERAS_BACKEND=tensorflow python $HOME/tierpsy-nn/tierpsy_nn/classify_strains/train_model.py \
--model_type 'resnet50' --is_wild_isolates True --saving_period 10 \
--sample_size_frames_s {sample_size} --sample_frequency_s {frequency}
## This tells the batch manager to execute the program cudaexecutable in the cuda directory of the users home directory.
'''

save_dir = '$WORK/run_scripts/classify_strains/grid_test'
if not os.path.exists:
    os.makedirs(save_dir)

save_str = save_dir + 'S{sample_size}_F{frequency:.2}_W_restnet50.sh'

sample_size_frames_s_dflt = 90
sample_frequency_s_dflt = 1/10

for sf in [1/30, 1/10, 1/3, 1., 3., 6.]:
    args = dict(sample_size = sample_size_frames_s_dflt, frequency=sf)
    cmd = cmd_str.format(**args)
    save_name = save_str.format(**args)
    print(save_name)
    
    with open(save_name, 'w') as fid:
        fid.write(cmd)
    
for ts in [15, 30, 60, 90, 120, 300, 600, 840]:
    args = dict(sample_size = ts, frequency=sample_frequency_s_dflt)
    cmd = cmd_str.format(**args)
    save_name = save_str.format(**args)
    print(save_name)
    
    with open(save_name, 'w') as fid:
        fid.write(cmd)
    