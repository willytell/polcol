#!/bin/bash

echo "begining..."

echo "Experiment: Results5-partC"
echo "Dataset: images from BBox"
echo "Configuration: without pre-trained weights, without data augmentation, without images for test, and saving weights on epoch end"

#echo "distA"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp0.py -a train -e 0 &>logs/no-test-distA-resnet50-with_bbox-exp0-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp1.py -a train -e 1 &>logs/no-test-distA-resnet50-with_bbox-exp1-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp2.py -a train -e 2 &>logs/no-test-distA-resnet50-with_bbox-exp2-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp3.py -a train -e 3 &>logs/no-test-distA-resnet50-with_bbox-exp3-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp4.py -a train -e 4 &>logs/no-test-distA-resnet50-with_bbox-exp4-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp5.py -a train -e 5 &>logs/no-test-distA-resnet50-with_bbox-exp5-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp6.py -a train -e 6 &>logs/no-test-distA-resnet50-with_bbox-exp6-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp7.py -a train -e 7 &>logs/no-test-distA-resnet50-with_bbox-exp7-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp8.py -a train -e 8 &>logs/no-test-distA-resnet50-with_bbox-exp8-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distA-resnet50-with_bbox-exp9.py -a train -e 9 &>logs/no-test-distA-resnet50-with_bbox-exp9-train


#echo "distB"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp0.py -a train -e 0 &>logs/no-test-distB-resnet50-with_bbox-exp0-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp1.py -a train -e 1 &>logs/no-test-distB-resnet50-with_bbox-exp1-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp2.py -a train -e 2 &>logs/no-test-distB-resnet50-with_bbox-exp2-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp3.py -a train -e 3 &>logs/no-test-distB-resnet50-with_bbox-exp3-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp4.py -a train -e 4 &>logs/no-test-distB-resnet50-with_bbox-exp4-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp5.py -a train -e 5 &>logs/no-test-distB-resnet50-with_bbox-exp5-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp6.py -a train -e 6 &>logs/no-test-distB-resnet50-with_bbox-exp6-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp7.py -a train -e 7 &>logs/no-test-distB-resnet50-with_bbox-exp7-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp8.py -a train -e 8 &>logs/no-test-distB-resnet50-with_bbox-exp8-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distB-resnet50-with_bbox-exp9.py -a train -e 9 &>logs/no-test-distB-resnet50-with_bbox-exp9-train
#

#echo "distC"
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp0.py -a train -e 0 &>logs/no-test-distC-resnet50-with_bbox-exp0-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp1.py -a train -e 1 &>logs/no-test-distC-resnet50-with_bbox-exp1-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp2.py -a train -e 2 &>logs/no-test-distC-resnet50-with_bbox-exp2-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp3.py -a train -e 3 &>logs/no-test-distC-resnet50-with_bbox-exp3-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp4.py -a train -e 4 &>logs/no-test-distC-resnet50-with_bbox-exp4-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp5.py -a train -e 5 &>logs/no-test-distC-resnet50-with_bbox-exp5-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp6.py -a train -e 6 &>logs/no-test-distC-resnet50-with_bbox-exp6-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp7.py -a train -e 7 &>logs/no-test-distC-resnet50-with_bbox-exp7-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp8.py -a train -e 8 &>logs/no-test-distC-resnet50-with_bbox-exp8-train
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distC-resnet50-with_bbox-exp9.py -a train -e 9 &>logs/no-test-distC-resnet50-with_bbox-exp9-train


echo "distD"
CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp0.py -a train -e 0 &>logs/no-test-distD-resnet50-with_bbox-exp0-train

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp1.py -a train -e 1 &>logs/no-test-distD-resnet50-with_bbox-exp1-train

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp2.py -a train -e 2 &>logs/no-test-distD-resnet50-with_bbox-exp2-train

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp3.py -a train -e 3 &>logs/no-test-distD-resnet50-with_bbox-exp3-train

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp4.py -a train -e 4 &>logs/no-test-distD-resnet50-with_bbox-exp4-train

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp5.py -a train -e 5 &>logs/no-test-distD-resnet50-with_bbox-exp5-train

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp6.py -a train -e 6 &>logs/no-test-distD-resnet50-with_bbox-exp6-train

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp7.py -a train -e 7 &>logs/no-test-distD-resnet50-with_bbox-exp7-train

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp8.py -a train -e 8 &>logs/no-test-distD-resnet50-with_bbox-exp8-train

CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_no-test-distD-resnet50-with_bbox-exp9.py -a train -e 9 &>logs/no-test-distD-resnet50-with_bbox-exp9-train


#echo "distE"


echo "finished!"
