#!/bin/bash


if [ $# -ne 1 ]; then
    echo "usage: <outdir>"
    exit 1
fi

outdir="$1"


if [ ! -f "$outdir"/.done ]; then
    # Create the output directory.
    mkdir -p "$outdir"


    echo "Extracting utterance ids for the training and testing set..."
    cat conf/clean_wavs.scp | \
         cut -d\  -f 1 > "$outdir"/keys || exit 1

    echo "Creating the 'scp' file for the clean data..."
    cp conf/clean_wavs.scp "$outdir"/wavs.scp || exit 1

    echo "Create phone list..."
    cp conf/phones "$outdir"/phones
    cat conf/phones | grep -v SIL > "$outdir"/phones_nosil

    echo "Prepare alignments..."
    cat conf/kaldi_ali.txt.gz | gunzip | \
        sed 's/\([[:upper:]]\)[0-9]/\1/g' | \
        sed s/_.\ /\ /g | \
        python utils/kaldi2mlf.py "$outdir"/ali.mlf

    echo "Splitting 'scp' files..."
    rm -fr "$outdir"/split
    mkdir -p "$outdir"/split
    cp "$outdir"/wavs.scp "$outdir"/split
    cd "$outdir"/split
    split -l 500 -d ./wavs.scp
    cd ../../
    date > "$outdir"/split/.done

    date > "$outdir"/.done

else
    echo "Data already prepared. Skipping."
fi

