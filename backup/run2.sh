#!/bin/bash

echo "begining..."

echo "Experiment: Results5-partC"
echo "Dataset: images from BBox"
echo "Configuration: without pre-trained weights, without data augmentation. To get probabilities of each image"

echo "dist1"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox.py -a train &>logs/dist1-resnet50-with_bbox-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox.py -a test  &>logs/dist1-resnet50-with_bbox-test

echo "dist2"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist2-resnet50-with_bbox.py -a train &>logs/dist2-resnet50-with_bbox-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist2-resnet50-with_bbox.py -a test  &>logs/dist2-resnet50-with_bbox-test

echo "dist3"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist3-resnet50-with_bbox.py -a train &>logs/dist3-resnet50-with_bbox-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist3-resnet50-with_bbox.py -a test  &>logs/dist3-resnet50-with_bbox-test

echo "dist4"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist4-resnet50-with_bbox.py -a train &>logs/dist4-resnet50-with_bbox-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist4-resnet50-with_bbox.py -a test  &>logs/dist4-resnet50-with_bbox-test


echo "dist5"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox.py -a train &>logs/dist5-resnet50-with_bbox-train
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5-resnet50-with_bbox.py -a test  &>logs/dist5-resnet50-with_bbox-test


echo "finished!"
