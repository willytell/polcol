#!/bin/bash
 
echo "=========================="

CNN='resnet50'

DIR='/home/master/tfm/ExperCNN/'
cd $DIR

for i in data*; do
   
   # list only directories
   END=$(ls -d $i/*/ | wc -l)
   
   for j in $(seq 1 $END); do
 
      cp /home/master/mcv-m5/code/config/cvc-classification-"$CNN"-from-scratch.py $DIR$i/cvc-"$i"-kfold"$j"-classification-"$CNN"-from-scratch.py
      sed -i '3s/.*/dataset_name                 = '"'"$i"-kfold$j'"'        # Dataset name/' $DIR$i/cvc-"$i"-kfold"$j"-classification-"$CNN"-from-scratch.py
       
      echo "CUDA_VISIBLE_DEVICES=0 python train.py -c config/cvc-"$i"-kfold"$j"-classification-"$CNN"-from-scratch.py \
-e "$CNN"-cvc-"$i"-kfold"$j"-from-scratch -s ~/datatmp/Datasets -l ~/datatmp/ &>logs/"$CNN"-cvc-"$i"-kfold"$j"-from-scratch.log" >> $DIR$i/run-"$i"-"$CNN".sh
  
    done
done

