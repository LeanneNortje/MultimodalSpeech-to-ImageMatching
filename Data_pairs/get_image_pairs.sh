#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script generates image pairs for each image in the specified dataset by using the smallest
# cosine distance between images to choose the pairs. Both pairs are from the same dataset.
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
keys=$pair_dir/keys.list
key_dir=$pair_dir/key_lists
key_pairs_dir=$pair_dir/key_pair_lists

if [ -f $keys ]; then
	rm -r $keys
fi

if [ -d $key_dir ]; then
	rm -r $key_dir
fi

if [ -d $key_pairs_dir ]; then
	rm -r $key_pairs_dir
fi

mkdir -p $pair_dir
./feature_keys.py $features_npz $keys
./divide_key_list.py $keys $key_dir $NUM_CORES
mkdir -p $key_pairs_dir
command="./distances.py --all_keys_list_fn $keys --this_keys_list_fn $key_dir/keys.JOB.list \
--feats_fn $features_npz --metric cosine --normalize False \
--key_pair_fn $key_pairs_dir/key_pairs.JOB.list --num_pairs $num_pairs"
$cmd 1 $NUM_CORES $key_pairs_dir/key_pairs.JOB.log "$command"


echo "Wait for jobs to finish before running image_pair_list.sh." 