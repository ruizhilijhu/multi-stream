#!/bin/bash


# Number of utterances for cross-validation.
nutts_cv=3741

if [ $# -ne 1 ]; then
    echo "usage: <outdir>"
    exit 1
fi

outdir="$1"


if [ ! -f "$outdir"/.done ]; then
    # Create the output directory.
    mkdir -p "$outdir"


    echo "Extracting utterance ids for the training and testing set..."
    nutts=$(cat conf/clean_wavs.scp | wc -l)
    head -n $((nutts - nutts_cv)) conf/clean_wavs.scp | \
         cut -d\  -f 1 > "$outdir"/train_keys || exit 1
    tail -n $((nutts_cv)) conf/clean_wavs.scp | \
         cut -d\  -f 1 > "$outdir"/test_keys || exit 1
    cat "$outdir"/{test,train}_keys > "$outdir"/all_keys || exit 1


    echo "Creating the 'scp' file for the clean data..."
    cp conf/clean_wavs.scp "$outdir"/wavs.scp || exit 1


    echo "Create phone list..."
    cp conf/phones "$outdir"/phones
    cat conf/phones | grep -v SIL > "$outdir"/phones_nosil

    echo "Prepare alignments..."
    cat conf/kaldi_ali.txt | \
        sed 's/\([[:upper:]]\)[0-9]/\1/g' | \
        sed s/_.\ /\ /g | \
        python utils/kaldi2mlf.py "$outdir"/ali.mlf

    date > "$outdir"/.done

else
    echo "Data already prepared. Skipping."
fi

