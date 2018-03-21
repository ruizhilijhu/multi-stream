#!/bin/bash


if [ $# -ne 4 ]; then
    echo "usage: <sge_opts> <context> <feadir> <outdir>"
    exit 1
fi

sge_opts="$1"
context="$2"
feadir="$3"
outdir="$4"

# Command to stack the features.
cmd="python utils/stack-features.py --context=$context $outdir"

# Number of jobs for the SGE.
njobs=$(ls "$feadir"/x* | wc -l)


if [ ! -f "$outdir"/.done ]; then
    echo "Stacking features..."
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

    date > "$outdir"/.done
else
    echo "Stacking features already done. Skipping."
fi

