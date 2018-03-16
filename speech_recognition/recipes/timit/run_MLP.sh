#!/bin/bash

. ./include_path.sh 

# Some Parameters 
nj_featgen=100
PHN_dir=PHN_files_dir
mkdir -p $PHN_dir
rm -rf $PHN_dir/*
phone_files=./phones.txt
feat_dir=fdlp_feats
mkdir -p $feat_dir
mkdir -p exp
command=queue.pl 

echo ==================
echo "Data Preparations"
echo ==================

for set in train test; do 
  scp_file=./data/$set/wav.scp
  echo "Processing" $set "PHN files"
  python ./local/cleanPhoneFiles.py $scp_file ./conf/phone.map $PHN_dir
done

# Split data for feature generation

for set in train test; do 
  scp_file=./data/$set/wav.scp
  split_dir=./data/$set/scp_splits
  mkdir -p $split_dir
  rm -rf $split_dir/*
  split_scps=""
  for n in $(seq $nj_featgen); do
    split_scps="$split_scps $split_dir/wav.$n.scp"
  done
  split_scp.pl $scp_file $split_scps
done 

echo =======================================
echo "Modulation Spectral Feature Generation"
echo =======================================

mkdir -p $feat_dir/log

echo "Computing FDLP features for Train and Test files..."

for set in train test; do 
  set=train
  split_dir=./data/$set/scp_splits
  $command -l arch=*64 -sync no --mem 5G --gpu 0 JOB=1:$nj_featgen \
    $feat_dir/log/fdlp_$set.JOB.log python -u ../../tools/modSpecFeatsCompute.py --get_phone_labels --only_center \
       $split_dir/wav.JOB.scp $feat_dir/$set.feats.JOB.npy ./conf/phone.map $PHN_dir &
done
