#!/bin/bash

# no-test-distA-resnet50-with_bbox-exp0-train
dist=$1   # dist=distA 
kfold=$2  # kfold=0

epochs[0]=$3
epochs[1]=$4
epochs[2]=$5
epochs[3]="${6}"
epochs[4]="${7}"
epochs[5]="${8}"
epochs[6]="${9}"
epochs[7]="${10}"
epochs[8]="${11}"
epochs[9]="${12}"

# in bash, to pass more than 10 parameters, one way is: "${10}"

for e in `seq 0 9`; do
   cad=no-test-"$dist"-resnet50-with_bbox-exp"$e"-train
   echo $cad
   exp_name=experiment"$e"_dataset"$kfold"_0_5_kfold"$kfold"
   echo $exp_name

   epoch=${epochs[e]} ;
   echo $epoch ;

   cat $cad | awk '/Init experiment: '$exp_name/',/Finish experiment: '$exp_name'/' > exp"$e"-kfold"$kfold";

   cat exp"$e"-kfold"$kfold" | awk '/Epoch '$epoch'/,/val_loss/' | awk '/y_pred/,/In epoch/'  >> stats-exp"$e".txt ;

   rm exp?-kfold?

done

paste -d, stats-exp0.txt stats-exp1.txt stats-exp2.txt stats-exp3.txt stats-exp4.txt stats-exp5.txt stats-exp6.txt stats-exp7.txt stats-exp8.txt stats-exp9.txt > results.txt
