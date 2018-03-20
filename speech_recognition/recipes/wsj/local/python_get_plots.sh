#!/bin/bash

run_cmd="python -u ../../analysis/plotTFFilters.py --featdim=$1 --time_samp=$2
$3 $4" ;
echo "## Running Command:"  $run_cmd " ##"
$run_cmd
echo "## Execution Ended ##"

[ -f $4 ] || exit 1; # Exit if output not found
