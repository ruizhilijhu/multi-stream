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


for n in 0 1 3 5 10 15; do
    context="$n"
    fdir="$feadir"/stacked_c${n}
    if [ ! -f "$fdir"/.done ]; then
        echo "Stacking features (c=${n})..."
        mkdir -p "$fdir"/log
        rm -f "$fdir"/log/jobarray*log
        njobs=$(ls "$feadir"/split/x* | wc -l)
        cmd="python utils/stack-features.py --context ${context} ${feadir}/raw ${fdir}"
        qsub -t 1-$njobs -sync y -cwd -j y -o "$fdir"/log/jobarray.\$TASK_ID.log \
            utils/jobarray.qsub "$cmd" "$feadir"/split || exit 1
        date > "$fdir"/.done
    else
        echo "Stacking already done. Skipping."
    fi


    if [ ! -f "$feadir"/central_frames_c${n}/.done ]; then
        echo "Extracting central frames (extra $n)..."
        mkdir -p "$feadir"/central_frames_c${n}

        cmd="python utils/extract-cf-frame.py \
                ${datadir}/phones ${datadir}/ali.mlf ${fdir} \
                ${feadir}/central_frames_c${n}/trainfea \
                ${feadir}/central_frames_c${n}/trainlab \
                ${datadir}/train_keys"
        qsub -l mem_free=100G,ram_free=100G -sync y -cwd -j y -o ${feadir}/central_frames_c${n}/sge.log \
            utils/job.qsub "$cmd" || exit 1

        python utils/extract-cf-frame.py \
            "$datadir"/phones "$datadir"/ali.mlf "$fdir" \
             "$feadir"/central_frames_c${n}/testfea \
             "$feadir"/central_frames_c${n}/testlab \
             "$datadir"/test_keys || exit 1

        date > "$feadir"/central_frames_c${n}/.done
    else
        echo "Extracting central frame already done. Skipping."
    fi
done

