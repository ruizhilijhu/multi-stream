#!/bin/bash

. ./include_path.sh

exp_name=experiment_1
ali=ali_dir
nj_ali=10
nj_featgen=500
PHN_dir=PHN_files_dir
mkdir -p $PHN_dir
phone_file=./phones.txt
feat_dir=fdlp_feats
command=queue.pl
fdlp_spectrum=true
<<skip
echo =================
echo "Data Preparation"
echo =================

python ./local/getPhoneFile.py $ali $nj_ali $PHN_dir $phone_file
echo "Data Preparation finshed..."
skip
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

feat_gen_options="--get_phone_labels"
nfilters=15
nmodulations=12
order=50
fduration=0.5
frate=100
around_center=0

mkdir -p $feat_dir
mkdir -p $feat_dir/log

if $fdlp_spectrum; then 
  echo "Computing FDLP spectral features for Train and Test files..."

  $command -l arch=*64 -sync no --mem 5G --gpu 0 JOB=51:$nj_featgen \
    ./fdlp_feats/log/fdlp_$set.JOB.log python -u  ../../tools/computeFDLPSpectralFeats.py $feat_gen_options \
      $split_dir/wav_$set.JOB.scp $feat_dir/$set.feats.JOB.npy \
        ./conf/phone.map $PHN_dir $nfilters $order $fduration $frate 
else

  echo "Computing FDLP modulation spectral features for Train and Test files..."

  $command -l arch=*64 -sync no --mem 5G --gpu 0 JOB=51:$nj_featgen \
    ./fdlp_feats/log/fdlp_$set.JOB.log python -u ../../tools/computeFDLPModSpecFeats.py $feat_gen_options \
           $split_dir/wav_$set.JOB.scp $feat_dir/$set.feats.JOB.npy \
              ./conf/phone.map $PHN_dir $nfilters $nmodulations $order $fduration $frate \
                  $around_center &
  $command -l arch=*64 -sync no --mem 5G --gpu 0 JOB=1:50 \
    ./fdlp_feats/log/fdlp_test.JOB.log python -u ../../tools/computeFDLPModSpecFeats.py $feat_gen_options \
          $split_dir/wav_test.JOB.scp $feat_dir/test.feats.JOB.npy \ 
              ./conf/phone.map $PHN_dir $nfilters $nmodulations $order $fduration $frate \
                  $around_center
fi

wait
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
  -l mem_free=50G,gpu=1,h=g01 -pe smp 1 -V ./local/python_run_MLP.sh $epoch_num $validation_rate $feat_dir $class_num $hid_num $node_num $model_out $log_dir


echo "Moving model and feature files out of code directory..."

move_exp_dir=../../experiments/$exp_name
mkdir -p $move_exp_dir
#mv $feat_dir $exp $move_exp_dir

rm -f $move_exp_dir/MLP_parameters

cat << EOF >> $move_exp_dir/MLP_parameters 
### MLP Details ### 
epoch_num=$epoch_num
validation_rate=$validation_rate
class_num=$class_num
hid_num=$hid_num
node_num=$node_num
model_out=$model_out
log_dir=$log_dir
EOF

rm -f $move_exp_dir/feature_parameters 

cat << EOF >> $move_exp_dir/feature_parameters 
### Feature Extraction Details ###
fdlp_spectrum=$fdlp_spectrum
feat_gen_options=$feat_gen_options
nfilters=$nfilters
nmodulations=$nmodulations
order=$order
fduration=$fduration
frate=$frate
around_center=$around_center
EOF


