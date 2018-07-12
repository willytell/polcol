#!/bin/bash

echo "begining..."

echo "Experiment: Results6-partA: dist5"
echo "Dataset: images from BBox"
echo "Configuration: without ImageNet and without data augmentation."

echo "dist5"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp0.py -a train -e 0 &>logs/dist5-resnet50-with_bbox-exp0-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp0.py -a test  -e 0 -k 2 &>logs/dist5-resnet50-with_bbox-exp0-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp1.py -a train -e 1 &>logs/dist5-resnet50-with_bbox-exp1-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp1.py -a test  -e 1 -k 2 &>logs/dist5-resnet50-with_bbox-exp1-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp2.py -a train -e 2 &>logs/dist5-resnet50-with_bbox-exp2-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp2.py -a test  -e 2 -k 4 &>logs/dist5-resnet50-with_bbox-exp2-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp3.py -a train -e 3 &>logs/dist5-resnet50-with_bbox-exp3-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp3.py -a test  -e 3 -k 4 &>logs/dist5-resnet50-with_bbox-exp3-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp4.py -a train -e 4 &>logs/dist5-resnet50-with_bbox-exp4-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp4.py -a test  -e 4 -k 2 &>logs/dist5-resnet50-with_bbox-exp4-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp5.py -a train -e 5 &>logs/dist5-resnet50-with_bbox-exp5-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp5.py -a test  -e 5 -k 4 &>logs/dist5-resnet50-with_bbox-exp5-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp6.py -a train -e 6 &>logs/dist5-resnet50-with_bbox-exp6-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp6.py -a test  -e 6 -k 4 &>logs/dist5-resnet50-with_bbox-exp6-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp7.py -a train -e 7 &>logs/dist5-resnet50-with_bbox-exp7-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp7.py -a test  -e 7 -k 3 &>logs/dist5-resnet50-with_bbox-exp7-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp8.py -a train -e 8 &>logs/dist5-resnet50-with_bbox-exp8-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp8.py -a test  -e 8 -k 1 &>logs/dist5-resnet50-with_bbox-exp8-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp9.py -a train -e 9 &>logs/dist5-resnet50-with_bbox-exp9-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-exp9.py -a test  -e 9 -k 4 &>logs/dist5-resnet50-with_bbox-exp9-test

echo "finished!"
