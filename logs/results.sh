#!/bin/bash

# $1: dist1-resnet50-with_bbox-imagenet-da-train
# $2: experiment number

e=$2
epochs=()

for f in 0 1 2 3 4; do  
   cad=experiment"$e"_dataset"$f"_24_5_kfold"$f"; 
   cat $1 | awk '/Init experiment: '$cad/',/Finish experiment: '$cad'/' > exp"$e"-fold"$f"; 
   best[f]="$(cat exp"$e"-fold"$f" |  grep "val_acc improved from" | tail -n1)"w; 
done

echo "val_loss, val_acc, f2-score, In-epoch, Norm Conf. Matrix"
for f in 0 1 2 3 4; do
    epoch="$(echo ${best[f]} | cut -d ':' -f1 | cut -d ' ' -f2)";
    #echo ">>> val_loss:";
    c1="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep val_loss | cut -d ':' -f4 | cut -d '-' -f1)"
    #echo ">>> val_acc:"
    c2="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep "val_acc:" | cut -d ':' -f5)"
    #echo ">>> f2-score:"
    c3="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep "f2-score:" | cut -d ':' -f2)"
    #echo ">>> In epoch:"
    c4="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep "In epoch:" | cut -d ':' -f2)"
    #echo ">>> Normalized Training Confusion Matrix:"
    c5="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | awk '/Normalized Training Confusion Matrix/ {for(i=1; i<=2; i++) {getline; print $0}}')"
    echo $c1' '$c2' '$c3' '$c4' '$c5
done    


echo "val_loss"
for f in 0 1 2 3 4; do
    epoch="$(echo ${best[f]} | cut -d ':' -f1 | cut -d ' ' -f2)";
    #echo ">>> val_loss:";
    c1="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep val_loss | cut -d ':' -f4 | cut -d '-' -f1)"
    echo $c1
done


echo "val_acc"
for f in 0 1 2 3 4; do
    epoch="$(echo ${best[f]} | cut -d ':' -f1 | cut -d ' ' -f2)";
    #echo ">>> val_acc:"
    c2="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep "val_acc:" | cut -d ':' -f5)"
    echo $c2
done 

echo "f2-score"
for f in 0 1 2 3 4; do
    epoch="$(echo ${best[f]} | cut -d ':' -f1 | cut -d ' ' -f2)";
    #echo ">>> f2-score:"
    c3="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep "f2-score:" | cut -d ':' -f2)"
    echo $c3
done 

 echo "In-epoch"
for f in 0 1 2 3 4; do
    epoch="$(echo ${best[f]} | cut -d ':' -f1 | cut -d ' ' -f2)";
    #echo ">>> In epoch:"
    c4="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep "In epoch:" | cut -d ':' -f2)"
    echo $c4
done 


echo "Norm Conf. Matrix"
for f in 0 1 2 3 4; do
    epoch="$(echo ${best[f]} | cut -d ':' -f1 | cut -d ' ' -f2)";
    #echo ">>> Normalized Training Confusion Matrix:"
    c5="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | awk '/Normalized Training Confusion Matrix/ {for(i=1; i<=2; i++) {getline; print $0}}')"
    echo $c5
done    



