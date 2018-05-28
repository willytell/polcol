#!/bin/bash
 
echo "=========================="

CNN='resnet50'

DIR='/home/master/tfm/ExperCNN/'
cd $DIR

for i in data*; do
   cp -p $DIR/$i/cvc-dataset*$CNN*.py /home/master/mcv-m5/code/config/
   cp -p $DIR/$i/run*$CNN*.sh /home/master/mcv-m5/code
   #cp -pr $DIR/$i/dataset* /home/master/datatmp/Datasets/classification
done

chmod +x /home/master/mcv-m5/code/run*$CNN*.sh

