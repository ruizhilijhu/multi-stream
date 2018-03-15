#!/bin/bash

. ./include_path.sh


ali=ali_dir
nj_ali=10
nj_featgen=500
PHN_dir=PHN_files_dir
mkdir -p $PHN_dir
phone_file=./phones.txt
feat_dir=fdlp_feats
command=queue.pl

echo =================
echo "Data Preparation"
echo =================

python ./local/getPhoneFile.py $ali $nj_ali $PHN_dir $phone_file
echo "Data Preparation finshed..."

echo =======================================
echo "Modulation Spectral Feature Generation"
echo =======================================

# Split data for feature generation 
scp_file=./data/wav.scp
split_dir=./data/scp_splits
set=train
mkdir -p $split_dir

split_scps=""
for n in $(seq $nj_featgen); do
  split_scps="$split_scps $split_dir/wav_$set.$n.scp"
done
split_scp.pl $scp_file $split_scps

# Keep 10% data for testing/development 

for n in $(seq 50); do 
  mv $split_dir/wav_$set.$n.scp $split_dir/wav_test.$n.scp
done 

# Extract Modulation Spectral features 
#
feat_gen_options="--get_phone_labels --only_center"
mkdir -p $feat_dir
mkdir -p $feat_dir/log

echo "Computing FDLP features for Train and Test files..."

$command -l arch=*64 -sync no --mem 5G --gpu 0 JOB=51:$nj_featgen \
  ./fdlp_feats/log/fdlp_$set.JOB.log python -u ../../tools/modSpecFeatsCompute.py --get_phone_labels --only_center \
         $split_dir/wav_$set.JOB.scp $feat_dir/$set.feats.JOB.npy ./conf/phone.map $PHN_dir &
$command -l arch=*64 -sync no --mem 5G --gpu 0 JOB=1:50 \
  ./fdlp_feats/log/fdlp_test.JOB.log python -u ../../tools/modSpecFeatsCompute.py --get_phone_labels --only_center \
        $split_dir/wav_test.JOB.scp $feat_dir/test.feats.JOB.npy ./conf/phone.map $PHN_dir

echo "Gather up all the data..."

python ./local/gatherAllData.py $feat_dir


echo ====================================
echo "Run simple MLP training and testing"
echo ====================================

mkdir -p ./exp/log

#$command -l arch=*64 -sync no --mem 50G --gpu 0  ./exp/log/mlp_train.log  python -u  ../../tools/trainMLPClassifier.py --epochs=20 --mvnorm $feat_dir 41 3 256 ./exp/final.mdl

epoch_num=20
validation_rate=200
class_num=41
hid_num=3
node_num=256
model_out=./exp/final.mdl
log_dir=./exp/log/mlp_train.log
qsub -cwd -j y -o $log_dir -e $log_dir -m eas -M sadhusamik@gmail.com \
  -l mem_free=50G,gpu=1,h=g01 -pe smp 1 -V ./python_run_MLP.sh $epoch_num $validation_rate $feat_dir $class_num $hid_num $node_num $model_out $log_dir
