#!/bin/bash

context=15

if [ $# -ne 2 ]; then
    echo "usage: <datadir> <feadir>"
    exit 1
fi

n_utt_per_job=500
datadir="$1"
feadir="$2"
scp="$datadir"/wavs.scp
keys="$datadir"/all_keys


# Create the output directory.
mkdir -p "$feadir"/{raw,stacked}


if [ ! -f "$feadir"/split/.done ]; then
    echo "Splitting 'scp' files..."
    rm -fr "$feadir"/split
    mkdir -p "$feadir"/split
    cp "$scp" "$feadir"/split
    cd "$feadir"/split
    split -l 500 -d ./wavs.scp
    cd ../../
    date > "$feadir"/split/.done
else
    echo "Splitting already done. Skipping."
fi


if [ ! -f "$feadir"/raw/.done ]; then
    echo "Extracting features..."
    mkdir -p "$feadir"/raw/log
    rm -f "$feadir"/raw/log/jobarray*log
    njobs=$(ls "$feadir"/split/x* | wc -l)
    qsub -t 1-$njobs -sync y -cwd -j y -o "$feadir"/raw/log/jobarray.\$TASK_ID.log \
        utils/jobarray.qsub "python utils/extract-features.py $feadir/raw" "$feadir"/split
    date > "$feadir"/raw/.done
else
    echo "Extracting fbank already done. Skipping."
fi


if [ ! -f "$feadir"/stacked/.done ]; then
    echo "Stacking features..."
    mkdir -p "$feadir"/stacked/log
    rm -f "$feadir"/stacked/log/jobarray*log
    cmd="python utils/stack-features.py --context ${context} ${feadir}/raw ${feadir}/stacked"
    qsub -t 1-$njobs -sync y -cwd -j y -o "$feadir"/stacked/log/jobarray.\$TASK_ID.log \
        utils/jobarray.qsub "$cmd" "$feadir"/split
        date > "$feadir"/stacked/.done
else
    echo "Stacking already done. Skipping."
fi


if [ ! -f "$feadir"/central_frames/.done ]; then
    echo "Extracting central frames..."
    mkdir -p "$feadir"/central_frames

    python utils/extract-cf-frame.py \
        "$datadir"/phones "$datadir"/ali.mlf "$feadir"/stacked \
         "$feadir"/central_frames/trainfea "$feadir"/central_frames/trainlab \
         "$datadir"/train_keys || exit 1
    python utils/extract-cf-frame.py \
        "$datadir"/phones "$datadir"/ali.mlf "$feadir"/stacked \
         "$feadir"/central_frames/testfea "$feadir"/central_frames/testlab \
         "$datadir"/test_keys || exit 1

    date > "$feadir"/central_frames/.done
else
    echo "Extracting central frame already done. Skipping."
fi
