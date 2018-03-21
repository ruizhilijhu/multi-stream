#!/bin/bash

context=15

if [ $# -ne 3 ]; then
    echo "usage: <sge_opts> <datadir> <outdir>"
    exit 1
fi

sge_opts="$1"
datadir="$2"
outdir="$3"
scp="$datadir"/wavs.scp
keys="$datadir"/all_keys

# Command to extract the features.
cmd="python utils/extract-features.py $outdir"


if [ ! -f "$outdir"/.done ]; then
    echo "Extracting features..."
    mkdir -p "$outdir"/log
    rm -f "$outdir"/log/*jobarray*log
    njobs=$(ls "$datadir"/split/x* | wc -l)
    qsub \
        -t 1-$njobs \
        -sync y \
        -cwd -j y \
        -o "$outdir"/log/jobarray.\$TASK_ID.log \
        $sge_opts \
        utils/jobarray.qsub "$cmd" "$datadir"/split || exit 1

    date > "$outdir"/.done

else
    echo "Features already extracted. Skipping."
fi

