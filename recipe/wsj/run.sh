#!/bin/bash

datadir=data
feadir=fbank
nnetdir=nnets

echo "================================================================"
echo "                   Data preparation                             "
echo "================================================================"
steps/prepare-data.sh "$datadir" || exit 1

echo "================================================================"
echo "                   Features extractions                         "
echo "================================================================"
#steps/extract-features.sh "$datadir" "$feadir" || exit 1
steps/extract-features-contexts.sh "$datadir" "$feadir" || exit 1


echo "================================================================"
echo "                   Neural network training                      "
echo "================================================================"
steps/train-nnet.sh "$datadir" "$feadir"  "$nnetdir" || exit 1
