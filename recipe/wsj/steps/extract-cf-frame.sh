#!/bin/bash


if [ $# -ne 4 ]; then
    echo "usage: <sge_opts> <datadir> <feadir> <outdir>"
    exit 1
fi

sge_opts="$1"
datadir="$2"
feadir="$3"
outdir="$4"

phones="$datadir"/phones
mlf="$datadir"/ali.mlf
keys="$datadir"/keys

# Command to stack the features.
cmd="python utils/extract-cf-frame.py  --exclude SIL \
    $phones $mlf $keys $outdir"

# Number of jobs for the SGE.
njobs=$(ls "$feadir"/x* | wc -l)

if [ ! -f "$outdir"/.done ]; then
    echo "Extracting central frame..."
    mkdir -p "$outdir"/log
    rm -f "$outdir"/log/jobarray*log

    qsub \
        -t 1-"$njobs" \
        -sync y \
        -cwd \
        -j y \
        -o "$outdir"/log/jobarray.\$TASK_ID.log \
        $sge_opts \
        utils/jobarray.qsub "$cmd" "$feadir" || exit 1

    ls "$outdir"/*npz > "$outdir"/list || exit 1

    echo "Mean normalization..."
    python utils/mnorm.py "$outdir"/list || exit 1

    date > "$outdir"/.done
else
    echo "Central frame already extracted. Skipping."
fi

