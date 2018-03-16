#!/bin/bash


gpu=`free-gpu`
echo "Free GPU device" $gpu 
python -u  ../../../tools/trainMLPClassifier.py --epochs=$1 --validation-rate=$2 --mvnorm $3 $4 $5 $6 $gpu $7
