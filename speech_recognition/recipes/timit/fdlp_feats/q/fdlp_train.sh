#!/bin/bash
cd /export/b18/ssadhu/python_scripts/recipes/timit
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
python -u ../../tools/modSpecFeatsCompute.py --get_phone_labels --only_center ./data/train/scp_splits/wav.${SGE_TASK_ID}.scp fdlp_feats/train.feats.${SGE_TASK_ID}.npy ./conf/phone.map PHN_files_dir 
EOF
) >fdlp_feats/log/fdlp_train.$SGE_TASK_ID.log
time1=`date +"%s"`
 ( python -u ../../tools/modSpecFeatsCompute.py --get_phone_labels --only_center ./data/train/scp_splits/wav.${SGE_TASK_ID}.scp fdlp_feats/train.feats.${SGE_TASK_ID}.npy ./conf/phone.map PHN_files_dir  ) 2>>fdlp_feats/log/fdlp_train.$SGE_TASK_ID.log >>fdlp_feats/log/fdlp_train.$SGE_TASK_ID.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>fdlp_feats/log/fdlp_train.$SGE_TASK_ID.log
echo '#' Finished at `date` with status $ret >>fdlp_feats/log/fdlp_train.$SGE_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch fdlp_feats/q/sync/done.135962.$SGE_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o fdlp_feats/q/fdlp_train.log -l arch=*64 -sync no -l mem_free=5G,ram_free=5G   -t 1:100 /export/b18/ssadhu/python_scripts/recipes/timit/fdlp_feats/q/fdlp_train.sh >>fdlp_feats/q/fdlp_train.log 2>&1
