#!/bin/bash

nfilters=100
nlayers=1
nunits=100
bsize=1000
lrate=1e-3
epochs=30


if [ $# -ne 3 ]; then
    echo "usage: <datadir> <feadir> <nnetdir>"
    exit 1
fi

datadir="$1"
feadir="$2"
nnetdir="$3"
extra="$4"
ntargets=$(cat "$datadir"/phones | wc -l)



for n in 0 1 3  5 10 15; do
    outdir=${nnetdir}/nnet_nf${nfilters}_nl${nlayers}_nu${nunits}_c${n}

    if [ ! -f "$outdir"/.done ]; then
        echo "Training neural network (extra ${n})..."
        mkdir -p "$outdir"

        cmd="python utils/train-nnet.py \
            --bsize $bsize --epochs $epochs --lrate $lrate --mnorm \
            "$feadir"/central_frames_c${n}/trainfea.npy \
            "$feadir"/central_frames_c${n}/trainlab.npy \
            "$feadir"/central_frames_c${n}/testfea.npy \
            "$feadir"/central_frames_c${n}/testlab.npy \
            $nfilters $ntargets $nlayers $nunits \
            "$outdir"/nnet.bin > "$outdir"/training.log"
        qsub -cwd -l mem_free=20G,ram_free=20G -sync y -j y -o "$toudir"/sge.log \
            utils/job.qsub "$cmd"


        date > "$outdir"/.done
    else
        echo "Neural network already trained. Skipping."
    fi

   if [ ! -f "$outdir"/filters_2D.html ]; then
       echo "Plotting neural network filters (extra ${n})..."
       mkdir -p "$outdir"
       python utils/plot-filters.py \
           "$outdir"/nnet.bin "$outdir"/filters_2D.html || exit 1
   else
       echo "Filters already plotted. Skipping."
   fi

done




