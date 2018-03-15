#!/bin/bash
cd /export/b18/ssadhu/python_scripts/recipes/wsj
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
-sync no --mem 50G --gpu 1 ./exp/log/mlp_train.log ./python_gpu_run.sh 20 200 fdlp_feats 41 3 256 ./exp/final.mdl 
EOF
) >arch=*64
time1=`date +"%s"`
 ( -sync no --mem 50G --gpu 1 ./exp/log/mlp_train.log ./python_gpu_run.sh 20 200 fdlp_feats 41 3 256 ./exp/final.mdl  ) 2>>arch=*64 >>arch=*64
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>arch=*64
echo '#' Finished at `date` with status $ret >>arch=*64
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.80930
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/arch=*64 -l hostname=g01    /export/b18/ssadhu/python_scripts/recipes/wsj/./q/arch=*64.sh >>./q/arch=*64 2>&1
