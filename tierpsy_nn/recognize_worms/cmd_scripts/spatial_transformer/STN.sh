
#!/bin/sh
#PBS -l walltime=24:00:00
## This tells the batch manager to limit the walltime for the job to XX hours, YY minutes and ZZ seconds.

#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1
## This tells the batch manager to use NN node with MM cpus and PP gb of memory per node with QQ gpus available.

module load anaconda3
module load cuda
## This job requires CUDA support.
source activate tierpsy

## copy temporary files
cp $WORK/recognize_worms/train_set/worm_ROI_samplesI.hdf5 $TMPDIR/

python /Users/ajaver/Documents/GitHub/tierpsy-nn/tierpsy_nn/recognize_worms/spatial_transformer/train.py --model_name STN --batch_size 64 --n_epochs 1000
## This tells the batch manager to execute the program cudaexecutable in the cuda directory of the users home directory.
