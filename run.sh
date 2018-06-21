#!/bin/bash

echo "begining..."

echo "====>"

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-imagenet-da.py -a train &>logs/dist1-resnet50-with_bbox-imagenet-da-train222222

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-imagenet-da.py -a test &>logs/dist1-resnet50-with_bbox-imagenet-da-test

# change range of experiments to train!!! 


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist2-resnet50-with_bbox-imagenet-da.py -a train &>logs/dist2-resnet50-with_bbox-imagenet-da-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist2-resnet50-with_bbox-imagenet-da.py -a test &>logs/dist2-resnet50-with_bbox-imagenet-da-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist3-resnet50-with_bbox-imagenet-da.py -a train &>logs/dist3-resnet50-with_bbox-imagenet-da-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist3-resnet50-with_bbox-imagenet-da.py -a test &>logs/dist3-resnet50-with_bbox-imagenet-da-test

##############


#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist2.py -a divide &>logs/dist2
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist3.py -a divide &>logs/dist3
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist4.py -a divide &>logs/dist4
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist5.py -a divide &>logs/dist5

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-da-class_weigths-imagenet.py -a train &>logs/dist1-resnet50-with_bbox-crop-da-class_weigths-imagenet-test

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-crop-da-class_weigths.py -a train &>logs/dist1-resnet50-with_bbox-crop-da-class_weigths-train

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-crop-da-class_weigths.py -a test &>logs/dist1-resnet50-with_bbox-crop-da-class_weigths-test


#echo "====>"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-without_augmentation.py -a train &>logs/dist1-resnet50-with_bbox-without_augmentation-train

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-without_augmentation.py -a test &>logs/dist1-resnet50-with_bbox-without_augmentation-test


#echo "====>"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-with_augmentation.py -a train &>logs/dist1-resnet50-with_bbox-with_augmentation-train

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-with_augmentation.py -a test &>logs/dist1-resnet50-with_bbox-with_augmentation-test


#echo "====>"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-with_augmentation_2flip.py -a train &>logs/dist1-resnet50-with_bbox-with_augmentation_2flip-train

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_dist1-resnet50-with_bbox-with_augmentation_2flip.py -a test &>logs/dist1-resnet50-with_bbox-with_augmentation_2flip-test


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
