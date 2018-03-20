#!/bin/bash

if $8; then 
  gpu=`free-gpu`
  echo "Free GPU device" $gpu
  run_cmd="python -u ../../tools/trainMLPClassifier.py --epochs=$1
  --validation-rate=$2 --gpu=$gpu --mvnorm --put_kink $3 $4 $5 $6 $7";
  echo "## Running Command:" $run_cmd " ##"
  $run_cmd
  echo "## Execution Ended ##"
 [ -f $7 ] || exit 1; # exit if model not found 
else
  run_cmd="python -u ../../../tools/trainMLPClassifier.py --epochs=$1
  --validation-rate$2 $9 $3 $4 $5 $6 $7" ; 
  echo "## Running Command:" $run_cmd " ##"
  $run_cmd
  echo "## Execution Ended ##"
fi
