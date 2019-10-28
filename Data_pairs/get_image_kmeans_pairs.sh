#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script generates image pairs with kmeans for each image in the specified dataset. Both pairs 
# are from the same dataset.
#

set -e

NUM_CORES=6
cmd=../Few_shot_learning/run.py
export PYTHONUNBUFFERED="YOUR_SET"

features_npz=$1
num_pairs=$2
if [ -z $features_npz ]; then 
    echo "Feature file required"
    exit 1
fi
if [ ! -f $features_npz ]; then
    echo "Feature file does not exist"
    exit 1
fi

regexp="|..\/\S+\/(\S+)\/(\S+)\.npz|"
if [[ $features_npz =~ $regexp ]]; then
	dataset=${BASH_REMATCH[1]}
	feats_name=${BASH_REMATCH[2]}
fi

pair_dir=$dataset/$feats_name
key_pair_file=$pair_dir/key_kmeans_pairs.list

if [ -f $key_pair_file ]; then
	rm -r $key_pair_file
fi


mkdir -p $pair_dir
./kmeans.py --feats_fn $features_npz --key_pair_fn $key_pair_file --num_pairs $num_pairs