#!/bin/bash 

# process the GTHistology.csv

Local='/home/willytell/Documentos/MCV/M9/TFM/ExperCNN'
Server='/home/master/tfm/ExperCNN'

DIR=$Local
#DIR=$Server

Dataset=$DIR/'GTHistologia.csv'

# outputs
Filenames=$DIR/'images_filenames.txt'
BinaryClassif=$DIR/'binary_classification.txt'
TernaryClassif=$DIR/'ternary_classfication.txt'

# remove the first line with headers and form the each image filename
#cat $Dataset | awk '{if (NR!=1) {print}}' | awk -F";" '{print $1}' | awk -F"_" '{print $3"_" $4 "_" $5".bmp"}' > $Filenames

cat $Dataset | awk '{if (NR!=1) {print}}' | cut -d ';' -f 1 | cut -d '_' -f 3- | awk '{print $1 ".bmp"}' > $Filenames

# the classification NEOPLASICO / NO NEOPLASICO
cat $Dataset | awk '{if (NR!=1) {print}}' | awk -F";" '{print $2}' | awk '{gsub("NO NEOPLASICO", "NONEOPLASICO"); print $1}' > $BinaryClassif

# the ternary classification ADENOMA / ASS / HIPERPLASICO
cat $Dataset | awk '{if (NR!=1) {print}}' | awk -F";" '{print $3}' > $TernaryClassif
