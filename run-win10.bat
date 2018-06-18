#!/bin/bash

echo "begining..."

echo "====>"

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-crop-da-francho.py -a train &>logs/dist1-resnet50-with_bbox-crop-da-francho-train


echo "finished!"
