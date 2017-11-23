#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:00:41 2017

@author: ajaver
"""
import os
from collections import OrderedDict


base_file = '''
#!/bin/sh
#PBS -l walltime={time_str}
## This tells the batch manager to limit the walltime for the job to XX hours, YY minutes and ZZ seconds.

#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1
## This tells the batch manager to use NN node with MM cpus and PP gb of memory per node with QQ gpus available.

module load anaconda3
module load cuda
## This job requires CUDA support.
source activate tierpsy

## copy temporary files
cp $WORK/recognize_worms/train_set/worm_ROI_samplesI.hdf5 $TMPDIR/

{cmd_str}
## This tells the batch manager to execute the program cudaexecutable in the cuda directory of the users home directory.
'''

if __name__ == '__main__':
    #add the parent directory to the log results
    pdir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
    save_dir = os.path.join(os.pardir, 'cmd_scripts', pdir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    time_str = '24:00:00'
    
    main_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
    
    dft_params = OrderedDict(
        model_name = None,
        batch_size = 64,
        n_epochs = 1000
    )

    all_exp = []
    reduced_options = [False, True]

    for mod in ['Simple', 'STN']:
        args = dft_params.copy()
        args['model_name'] = mod
        all_exp.append(args)


    short_add = OrderedDict(
        model_name = lambda x : x
    )
    
    for args in all_exp:
        args_d = ' '.join(['--{} {}'.format(*d) for d in args.items()])
        cmd_str = 'python {} {}'.format(main_file, args_d)
        
        
        f_content = base_file.format(time_str=time_str, 
                              cmd_str=cmd_str
                              )

        f_name = [func(args[k]) for k,func in short_add.items()]
        f_name = '_'.join([x for x in f_name if x]) + '.sh'
        f_name = os.path.join(save_dir, f_name)
        
        with open(f_name, 'w') as fid:
            fid.write(f_content)
        