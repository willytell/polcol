#!/bin/bash

echo "begining..."

echo "====>"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-without_augmentation.py -a train &>logs/dist1-resnet50-with_bbox-without_augmentation


echo "====>"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-with_augmentation.py -a train &>logs/dist1-resnet50-with_bbox-with_augmentation



#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-vgg.py -a train &>logs/vgg-from-scratch-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-vgg.py -a test &>logs/vgg-from-scratch-test

#echo "====>"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-resnet50.py -a train &>logs/resnet50-from-scratch-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-resnet50.py -a test &>logs/resnet50-from-scratch-test
#
#
#echo "====>"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-vgg-with_bbox.py -a train &>logs/vgg-from-scratch-bbox-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-vgg-with_bbox.py -a test &>logs/vgg-from-scratch-bbox-test
#
#echo "====>"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-resnet50-with_bbox.py -a train &>logs/resnet50-from-scratch-bbox-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-resnet50-with_bbox.py -a test &>logs/resnet50-from-scratch-bbox-test
#
#
#echo "====>"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-vgg-with_recortadas_inpainting.py -a train &>logs/vgg-from-scratch-recortadas-inpainting-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-vgg-with_recortadas_inpainting.py -a test &>logs/vgg-from-scratch-recortadas-inpainting-test
#
#echo "====>"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-resnet50-with_recortadas_inpainting.py -a train &>logs/resnet50-from-scratch-recortadas_inpainting-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_exp1-resnet50-with_recortadas_inpainting.py -a test &>logs/resnet50-from-scratch-recortadas_inpainting-test

echo "finished!"
