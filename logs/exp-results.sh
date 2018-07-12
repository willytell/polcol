#!/bin/bash

# example: 
#     rm c?.txt ; rm exp?-fold? ; rm results.txt
#     ./exp-results.sh dist4-resnet50-with_bbox-imagenet-exp9-train 9 38 26 14 41 41
#     cat results.txt

fname=$1
e=$2   # experiment number

best[0]=$3
best[1]=$4
best[2]=$5
best[3]=$6
best[4]=$7


# c is the number of columns to print
for c in 1 2 3 4 5;do
       epochs=()

       for f in 0 1 2 3 4; do  
          cad=experiment"$e"_dataset"$f"_24_5_kfold"$f";  # experiment0_dataset0_24_5_kfold0
          cat $fname | awk '/Init experiment: '$cad/',/Finish experiment: '$cad'/' > exp"$e"-fold"$f"; 
          #best[f]="$(cat exp"$e"-fold"$f" |  grep "val_acc improved from" | tail -n1)"w; 
       done
       
       if [ $c -eq 1 ]
       then
          # echo "val_loss"
          for f in 0 1 2 3 4; do
             epoch=${best[f]};
             #echo ">>> val_loss:";
             c1="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep val_loss | cut -d ':' -f4 | cut -d '-' -f1)"
             echo $c1 >> c"$c".txt
          done
       fi
       
       if [ $c -eq 2 ]
       then
          # echo "val_acc"
          for f in 0 1 2 3 4; do
              epoch=${best[f]};
              #echo ">>> val_acc:"
              c2="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep "val_acc:" | cut -d ':' -f5)"
              echo $c2 >> c"$c".txt
          done 
       fi
          
       if [ $c -eq 3 ]
       then
          # echo "f2-score"
          for f in 0 1 2 3 4; do
              epoch=${best[f]};
              #echo ">>> f2-score:"
              c3="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep "f2-score:" | cut -d ':' -f2)"
              echo $c3 >> c"$c".txt
          done 
       fi
       
          
       if [ $c -eq 4 ]
       then
          # echo "In-epoch"
          for f in 0 1 2 3 4; do
              epoch=${best[f]};
              #echo ">>> In epoch:"
              c4="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | grep "In epoch:" | cut -d ':' -f2)"
              echo $c4 >> c"$c".txt
          done 
       fi   
       
          
       if [ $c -eq 5 ]
       then
          # echo "Norm Conf. Matrix"
          for f in 0 1 2 3 4; do
              epoch=${best[f]};
              #echo ">>> Normalized Training Confusion Matrix:"
              c5="$(cat exp"$e"-fold"$f" | awk '/Epoch '$epoch'/,/val_loss/' | awk '/Normalized Training Confusion Matrix/ {for(i=1; i<=2; i++) {getline; print $0}}')"
              echo $c5 >> c"$c".txt
          done    
       fi

       echo " " >> c"$c".txt
 
    done
          
paste -d, c1.txt c2.txt c3.txt c4.txt c5.txt > results.txt          
            
