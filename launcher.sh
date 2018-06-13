#!/bin/bash

CONFIG_PATH="/imatge/mgorriz/work/Guillermo/polcol/config"
OUTPUT_PATH="/imatge/mgorriz/work/Guillermo/polcol/logs"

if [ "$1" = "divide" ]
then
    srun --pty --gres=gpu:1,gmem:12G --mem 8G python3 #main.py -c $CONFIG_PATH/conf_exp1-sm.py -a divide #&>logs/vgg-from-scratch-generate
elif [ "$1" = "train" ]
then
    srun --pty --gres=gpu:1,gmem:12G --mem 8G python3 main.py -c $CONFIG_PATH/conf_dist1-resnet50-with_bbox-DEBUGGIN.py -a train #&>logs/vgg-from-scratch-train 

elif [ "$1" = "test" ]
then
    srun --pty --gres=gpu:1,gmem:12G --mem 8G python3 main.py -c config/conf_dist1-resnet50-with_bbox-DEBUGGIN.py -a test #&>logs/vgg-from-scratch-test
else
    echo "Unknown option; expected: divide, train or test."
fi
