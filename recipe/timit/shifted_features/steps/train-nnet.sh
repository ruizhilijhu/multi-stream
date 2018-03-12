#!/bin/bash

nfilters=64
nlayers=3
nunits=256
bsize=1000
lrate=1e-3
epochs=15


if [ $# -ne 3 ]; then
    echo "usage: <datadir> <feadir> <nnetdir>"
    exit 1
fi

datadir="$1"
feadir="$2"
nnetdir="$3"
ntargets=$(cat "$datadir"/phones | wc -l)

outdir=${nnetdir}/nnet_nf${nfilters}_nl${nlayers}_nu${nunits}


if [ ! -f "$outdir"/.done ]; then
    echo "Training neural network..."
    mkdir -p "$outdir"

    python utils/train-nnet.py \
        --bsize $bsize --epochs $epochs --lrate $lrate --mvnorm \
        "$feadir"/central_frames/trainfea.npy \
        "$feadir"/central_frames/trainlab.npy \
        "$feadir"/central_frames/testfea.npy \
        "$feadir"/central_frames/testlab.npy \
        $nfilters $ntargets $nlayers $nunits \
        "$outdir"/nnet.bin || exit 1

    date > "$outdir"/.done
else
    echo "Neural network already trained. Skipping."
fi

if [ ! -f "$outdir"/filters_2D.html ]; then
    echo "Plotting neural network filters..."
    mkdir -p "$outdir"
    python utils/plot-filters.py \
        "$outdir"/nnet.bin "$outdir"/filters_2D.html || exit 1
else
    echo "Filters already plotted. Skipping."
fi

