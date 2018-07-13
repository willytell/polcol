#!/bin/bash

echo "begining..."

echo "Experiment: Results6-partC: dist5"
echo "Goal: save the images that feed the CNN in a directory and later add the prefix ranking to each image."
echo ""
echo "Dataset: images from BBox"
echo "Configuration: without ImageNet and without data augmentation."

echo "dist5"
#python save_prefix_image.py -i "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/BBox" -d "data.csv" -o "/home/willytell/Experiments/feed_cnn" -a "feed_cnn"

#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_feed_cnn-resnet50-with_bbox.py -a train -e 0 -k 0 #&>logs/feed_cnn-resnet50-with_bbox-exp0-kfold0-train
#CUDA_VISIBLE_DEVICES=0 python3 main.py -c config/conf_feed_cnn-resnet50-with_bbox.py -a test  -e 0 -k 0 #&>logs/feed_cnn-resnet50-with_bbox-exp0-kfold0-test

# Option -c: 
#    -c 1 = "% Aciertos" 
#    -c 2 = "Mean (Diff)"
#    -c 3 = "DevStd (Diff)"
#python3 save_prefix_image.py -i "/home/willytell/Experiments/feed_cnn/exp0-kfold0/feeding_images-test/all" -d "data.csv" -c 1 -o "/home/willytell/Experiments/feed_cnn/exp0-kfold0/feeding_images-test/all-prefix-aciertos" -a "prefix"

python3 save_prefix_image.py -i "/home/willytell/Experiments/feed_cnn/exp0-kfold0/feeding_images-test/all" -d "data.csv" -c 2 -o "/home/willytell/Experiments/feed_cnn/exp0-kfold0/feeding_images-test/all-prefix-mean" -a "prefix"

python3 save_prefix_image.py -i "/home/willytell/Experiments/feed_cnn/exp0-kfold0/feeding_images-test/all" -d "data.csv" -c 3 -o "/home/willytell/Experiments/feed_cnn/exp0-kfold0/feeding_images-test/all-prefix-devstd" -a "prefix"


echo "finished!"
