
#!/bin/bash
# Run FDLP feature generation 

set=$1;
nj=$2;

# Generate the FDLP features in the same format as Kaldi 

scp_file=/export/b18/ssadhu/kaldi/egs/timit/s5/data/$set/wav.scp
split_dir=/export/b18/ssadhu/python_scripts/scp_splits 
mkdir -p $split_dir 

# Split data into multiple parts 

split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $split_dir/wav_$set.$n.scp"
  done
/export/b18/ssadhu/kaldi/egs/timit/s5/utils/split_scp.pl $scp_file $split_scps

/export/b18/ssadhu/kaldi/egs/timit/s5/utils/queue.pl -l arch=*64 -sync no --mem 5G --gpu \
0 JOB=1:$nj ./log_$set/FDLP.JOB.log python -u  modSpecCompute.py \
  --get_phone_labels --only_center --use_38_phones $split_dir/wav_$set.JOB.scp \
./outdir ./phones.60-48-39.map 




