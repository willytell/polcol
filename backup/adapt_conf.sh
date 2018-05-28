#!/bin/bash 

#echo "hola mundo" $1 $2 $3 $4 $5 $6 $7
# dataset_directory, working_directory, "/config.py", kfold+str(i),str(len(train_index)), str(len(validation_index)), str(n_images_for_test)])

cp ./$3 $2/$4


sed -i '5s/.*/dataset_name    = '"'$4'"'   # Name of the dataset/' $2/$4/$3

sed -i '10s/.*/n_images_train  = '$5'/' $2/$4/$3

sed -i '12s/.*/n_images_valid  = '$6'/' $2/$4/$3

sed -i '11s/.*/n_images_test   = '$7'/' $2/$4/$3


# config.py
#
#  5 dataset_name    = 'kfold12'   # Name of the dataset
#  6 n_channels      = 3               # Number of channels of the images^M
#  7 color_mode      = 'rgb'           # Number of channels of the images [rgb, grayscale]^M
#  8 void_class      = []              # Labels of the void classes (It should be greather than the number of classes)^M
#  9 img_shape       = (1920, 1080)    # Shape of the input image (Height, Width)^M
# 10 n_images_train  = 186             # Number of training images^M
# 11 n_images_test   = 22              # Number of testing images^M
# 12 n_images_valid  = 18              # (!!!) BTSC has no official validation set. We use test here, but do not overfit it!^M
# 13 class_mode      = 'categorical'   # {'categorical', 'binary', 'sparse', 'segmentation', None}:^M

