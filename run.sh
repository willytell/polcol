#!/bin/bash

echo "begining..."

echo "====>"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-vgg.py -a test &>logs/vgg-from-scratch-test

echo "====>"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-resnet50.py -a test &>logs/resnet50-from-scratch-test


echo "====>"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-vgg-with_bbox.py -a test &>logs/vgg-from-scratch-bbox-test

echo "====>"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-resnet50-with_bbox.py -a test &>logs/resnet50-from-scratch-bbox-test

echo "finished!"
