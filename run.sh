#!/bin/bash

echo "begining..."

echo "Experiment: Results5-partB"
echo "Dataset: images from BBox"
echo "Configuration: ImageNet pre-trained weights, without data augmentation."

echo "dist1"
#python3 main.py -c config/conf_dist1-resnet50-with_bbox-DEBUGGIN.py -a train &>logs/dist1-resnet50-with_bbox-DEBUGGIN-train
python3 main.py -c config/conf_dist1-resnet50-with_bbox-DEBUGGIN-2.py -a train &>logs/dist1-resnet50-with_bbox-DEBUGGIN-2-train
#python3 main.py -c config/conf_dist1-resnet50-with_bbox-imagenet.py -a test  &>logs/dist1-resnet50-with_bbox-imagenet-test

#echo "dist2"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist2-resnet50-with_bbox-imagenet.py -a train &>logs/dist2-resnet50-with_bbox-imagenet-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist2-resnet50-with_bbox-imagenet.py -a test  &>logs/dist2-resnet50-with_bbox-imagenet-test
#
#echo "dist3"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist3-resnet50-with_bbox-imagenet.py -a train &>logs/dist3-resnet50-with_bbox-imagenet-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist3-resnet50-with_bbox-imagenet.py -a test  &>logs/dist3-resnet50-with_bbox-imagenet-test
#
#echo "dist4"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist4-resnet50-with_bbox-imagenet.py -a train &>logs/dist4-resnet50-with_bbox-imagenet-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist4-resnet50-with_bbox-imagenet.py -a test  &>logs/dist4-resnet50-with_bbox-imagenet-test
#
#
#echo "dist5"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-imagenet.py -a train &>logs/dist5-resnet50-with_bbox-imagenet-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox-imagenet.py -a test  &>logs/dist5-resnet50-with_bbox-imagenet-test


echo "finished!"
