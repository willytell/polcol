#!/bin/bash

echo "begining..."

echo "Experiment: Results6-partE"
echo "Dataset: images from BBox"
echo "Configuration: withou ImageNet and without data augmentation"

echo "dist50"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp0.py -a train -e 0 &>logs/dist50-resnet50-bbox-exp0-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp0.py -a test  -e 0 -k 2 &>logs/dist50-resnet50-bbox-exp0-test


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp1.py -a train -e 1 &>logs/dist50-resnet50-bbox-exp1-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp1.py -a test  -e 1 -k 0 &>logs/dist50-resnet50-bbox-exp1-test


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp2.py -a train -e 2 &>logs/dist50-resnet50-bbox-exp2-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp2.py -a test  -e 2 -k 2 &>logs/dist50-resnet50-bbox-exp2-test


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp3.py -a train -e 3 &>logs/dist50-resnet50-bbox-exp3-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp3.py -a test  -e 3 -k 4 &>logs/dist50-resnet50-bbox-exp3-test


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp4.py -a train -e 4 &>logs/dist50-resnet50-bbox-exp4-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp4.py -a test  -e 4 -k 0 &>logs/dist50-resnet50-bbox-exp4-test


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp5.py -a train -e 5 &>logs/dist50-resnet50-bbox-exp5-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp5.py -a test  -e 5 -k 0 &>logs/dist50-resnet50-bbox-exp5-test


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp6.py -a train -e 6 &>logs/dist50-resnet50-bbox-exp6-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp6.py -a test  -e 6 -k 2 &>logs/dist50-resnet50-bbox-exp6-test


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp7.py -a train -e 7 &>logs/dist50-resnet50-bbox-exp7-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp7.py -a test  -e 7 -k 4 &>logs/dist50-resnet50-bbox-exp7-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp8.py -a train -e 8 &>logs/dist50-resnet50-bbox-exp8-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp8.py -a test  -e 8 -k 4 &>logs/dist50-resnet50-bbox-exp8-test


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp9.py -a train -e 9 &>logs/dist50-resnet50-bbox-exp9-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist50-resnet50-bbox-exp9.py -a test  -e 9 -k 3 &>logs/dist50-resnet50-bbox-exp9-test


echo "finished!"
