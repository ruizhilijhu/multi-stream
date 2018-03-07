#!/bin/bash

datadir=data
feadir=fbank
nnetdir=nnets
timit_root=/export/corpora/LDC/LDC93S1/timit/TIMIT

echo "================================================================"
echo "                   Data preparation                             "
echo "================================================================"
steps/prepare-data.sh "$timit_root" conf/test_speakers "$datadir" || exit 1


echo "================================================================"
echo "                   Features extractions                         "
echo "================================================================"
steps/extract-features.sh "$datadir" "$feadir" || exit 1


echo "================================================================"
echo "                   Neural network training                      "
echo "================================================================"
steps/train-nnet.sh "$datadir" "$feadir"  "$nnetdir" || exit 1
