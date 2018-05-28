#!/bin/bash
 
echo "=========================="
CNN='resnet50'

cd '/home/master/mcv-m5/code/'
rm run*$CNN*.sh

cd '/home/master/mcv-m5/code/config/'
rm cvc-dataset*$CNN*.py
rm cvc-dataset*$CNN*.pyc

cd '/home/master/mcv-m5/code/logs'
rm "$CNN"-cvc-dataset*.log

DIR='/home/master/tfm/ExperCNN/'
cd $DIR

for i in data*; do
   rm $i/cvc-dataset*$CNN*.py
   rm $i/run*$CNN*.sh
done

rm /home/master/TensorBoardLogs/*
rm -rf /home/master/TensorBoardLogs-*$CNN*

cd '/home/master/datatmp/master/Experiments/'
for e in dataset*; do
   rm -rf $e/"$CNN"-*
done
