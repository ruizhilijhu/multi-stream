#!/bin/bash

noise_level=-30
lowfreq=1000
highfreq=1500


if [ $# -ne 3 ]; then
    echo "usage: <timit_root> <test_speaker_list> <outdir>"
    exit 1
fi

timit_root="$1"
test_speaker_list="$2"
outdir="$3"
wavdir="$outdir"/noisy_data_snr${noise_level}_lf${lowfreq}_hf${highfreq}


# Check if the step has already been done.
if [ -f "$outdir"/.done ]; then
    echo "Data already prepared. Skipping."
    exit 0
fi


# Create the output directory.
mkdir -p "$outdir"


echo "Extracting utterance ids for the training and testing set..."
find "$timit_root" -iwholename '*train*wav' -not \( -iname 'sa*' \) | \
    python utils/create-keys.py --depth 1 > "$outdir"/train_keys || exit 1
find "$timit_root" -iwholename '*test*wav' -not \( -iname 'sa*' \) | \
     python utils/create-keys.py --depth 1 | \
    grep -f "$test_speaker_list"  > "$outdir"/test_keys || exit 1
cat "$outdir"/{test,train}_keys > "$outdir"/all_keys || exit 1


echo "Creating the 'scp' file the clean data..."
find "$timit_root" -iwholename '*train*wav' -not \( -iname 'sa*' \) | \
    python utils/create-keys.py --depth 1 --echo | \
    awk '{print $1 " sph2pipe -f wav " $2 " |"}' > "$outdir"/clean_wavs.scp || exit 1
find "$timit_root" -iwholename '*test*wav' -not \( -iname 'sa*' \) | \
    python utils/create-keys.py --depth 1 --echo | \
    grep -f "$test_speaker_list" | \
    awk '{print $1 " sph2pipe -f wav " $2 " |"}'>> "$outdir"/clean_wavs.scp || exit 1


if [ ! -f "$wavdir"/.done ]; then
    echo "Creating noisy speech..."
    mkdir -p "$wavdir"
    python utils/add-noise.py  "$wavdir" "$outdir"/clean_wavs.scp \
        --low-freq $lowfreq --high-freq $highfreq --noise-level $noise_level || exit 1
    date > "$wavdir"/.done
else
    echo "Noisy speech already generated. Skipping."
fi


echo "Creating the 'scp' file the noisy data..."
find "$wavdir" -iname '*wav' | python utils/create-keys.py --echo | \
    awk '{print $1 " " $2}' > "$outdir"/wavs.scp || exit 1


echo "Create phone list..."
cat conf/phones.60-48-39.map | grep -v q | awk '{print $1 " " $3}' > "$outdir"/phone_map
cat "$outdir"/phone_map | grep -v sil | awk '{print $2}' | sort | uniq > "$outdir"/phones


echo "Create alignments..."
python utils/prepare-timit-ali.py "$timit_root/*/*/*/*PHN" > "$outdir"/ali.mlf


date > "$outdir"/.done

