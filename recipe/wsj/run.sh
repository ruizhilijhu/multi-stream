#!/bin/bash

#######################################################################
## CONFIGURATION

# Directory structure
datadir=data
feadir=fbank
nnetdir=nnets

# How many frames to stack together for a single features frame.
context=15

# SGE options.
fea_sge_opts="-tc 20 -l mem_free=0.5G,ram_free=0.5G"
train_sge_opts="-l gpu=1,mem_free=1G,ram_free=1G,hostname=b02"

#######################################################################

echo "================================================================"
echo "                   Data preparation                             "
echo "================================================================"
local/prepare-data.sh "$datadir" || exit 1

echo "================================================================"
echo "                   Features extractions                         "
echo "================================================================"
steps/extract-features.sh "$fea_sge_opts" "$datadir" "$feadir" || exit 1
steps/stack-features.sh "$fea_sge_opts" "$context" "$feadir" \
    "$feadir/stacked_c${context}" || exit 1
steps/extract-cf-frame.sh "$fea_sge_opts" "$datadir" \
    "$feadir/stacked_c${context}" "$feadir/cf_c${context}" || exit 1


echo "================================================================"
echo "                   Neural network training                      "
echo "================================================================"
steps/train-nnet.sh "$train_sge_opts" "$datadir" "$feadir/cf_c${context}"  "$nnetdir" || exit 1
