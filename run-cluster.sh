#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /home/csanchez/polcol # working directory
#SBATCH -t 5-00:05 # Runtime in D-HH:MM
#SBATCH -p dcc # Partition to submit to
#SBATCH --mem 16384 # 16GB solicitados.
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

exp=$1
kfold=$2

echo "begining..."

echo "Experiment: Results6-partA"
echo "Dataset: images from BBox"
echo "Configuration: without ImageNet and without data augmentation. To compare with Results5-PartA"

echo "dist5, experiment $exp, kfold $kfold"

python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp"$exp".py -a train -e $exp -k $kfold &>logs/dist5-resnet50-with_bbox-exp"$exp"-kfold"$kfold"-train
#python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp0.py -a test  -e 0 -k 4 &>logs/dist5-resnet50-with_bbox-exp0-test

#echo "dist3"
#python3 main.py -c config/conf_dist3-resnet50-with_bbox-without-imagenet-with-da-exp"$exp".py -a train -e $exp -k $kfold &>logs/dist3-resnet50-with_bbox-without-imagenet-with-da-exp"$exp"-kfold"$kfold"-train
#python3 main.py -c config/conf_dist3-resnet50-with_bbox-without-imagenet-with-da-exp0.py -a test  -e 0 -k 4 &>logs/dist3-resnet50-with_bbox-without-imagenet-with-da-exp0-test

echo "finished!"
