#!/bin/bash
cd /export/b18/ssadhu/python_scripts/recipes/timit
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
fdlp_feats/log/fdlp_train.JOB.log python -u ../../tools/modSpecFeatsCompute.py --get_phone_labels --only_center ./data/test/scp_splits/wav_train.JOB.scp fdlp_feats/train.feats.JOB.npy ./conf/phone.map PHN_files_dir 
EOF
) >JOB=1:
time1=`date +"%s"`
 ( fdlp_feats/log/fdlp_train.JOB.log python -u ../../tools/modSpecFeatsCompute.py --get_phone_labels --only_center ./data/test/scp_splits/wav_train.JOB.scp fdlp_feats/train.feats.JOB.npy ./conf/phone.map PHN_files_dir  ) 2>>JOB=1: >>JOB=1:
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>JOB=1:
echo '#' Finished at `date` with status $ret >>JOB=1:
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.123926
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/JOB=1: -l arch=*64 -sync no  -l mem_free=5G,ram_free=5G   /export/b18/ssadhu/python_scripts/recipes/timit/./q/JOB=1:.sh >>./q/JOB=1: 2>&1
