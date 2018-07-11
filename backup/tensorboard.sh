#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /home/csanchez/polcol # working directory
#SBATCH -t 5-00:05 # Runtime in D-HH:MM
#SBATCH -p dcc # Partition to submit to
#SBATCH --mem 16384 # 16GB solicitados.
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
# #SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
# #SBATCH -e %x_%u_%j.err # File to which STDERR will be written


cd ~/Experiments
tensorboard --logdir=cluster:dist5/resnet50-from-scratch-bbox-imagenet-exp0/TensorBoard-experiment0/ --port 6099

