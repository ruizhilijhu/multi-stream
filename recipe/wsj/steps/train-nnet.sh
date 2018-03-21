#!/bin/bash

feadim=930
nfilters=64
nlayers=3
nunits=256
bsize=1000
lrate=1e-3
epochs=30


if [ $# -ne 4 ]; then
    echo "usage: <sge_opts> <datadir> <feadir> <nnetdir>"
    exit 1
fi

sge_opts="$1"
datadir="$2"
feadir="$3"
nnetdir="$4"
ntargets=$(cat "$datadir"/phones | wc -l)

outdir=${nnetdir}/nnet_nf${nfilters}_nl${nlayers}_nu${nunits}


if [ ! -f "$outdir"/.done ]; then
    echo "Creating neural network..."
    mkdir -p "$outdir"

    python utils/create-2dF-nnet.py \
        $feadim $ntargets $nfilters $nlayers $nunits "$outdir"/init_nnet.bin || exit 1

    echo "Training neural network..."
    mkdir -p "$outdir"/log
    rm -f "$outdir"/log/*log
    cmd="python utils/train-nnet.py --gpu --validation-rate 1000 --epochs=${epochs} --lrate ${lrate} --bsize ${bsize} $feadir/list $outdir/init_nnet.bin  $outdir/nnet.bin"
    qsub \
        -sync y \
        -cwd \
        -j y \
        -o "$outdir"/log/training.log \
        $sge_opts \
        utils/job.qsub "$cmd"

    #qsub \
    #    -sync y \
    #    -cwd \
    #    -j y \
    #    -o "$outdir"/log/training.log \
    #    $sge_opts \
    #    utils/job.qsub "echo hello, world" || exit 1
    #    #utils/job.qsub "$cmd" || exit 1


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

