#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script concatenates the N generated speech pair scripts into one file. 
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

regexp="|..\/\S+\/(\S+)\/(\S+)\/(\S+)\/(\S+)\/(\S+)\.npz|"
if [[ $features_npz =~ $regexp ]]; then
	dataset=${BASH_REMATCH[1]}
	feats_or_subset=${BASH_REMATCH[2]}
	raw_or_segments=${BASH_REMATCH[3]}
	feats_type=${BASH_REMATCH[4]}
	feats_name=${BASH_REMATCH[5]}
fi

pair_dir=$dataset/$feats_or_subset/$raw_or_segments/$feats_type/$feats_name
key_pairs_dir=$pair_dir/key_pair_lists
key_pair_file=$pair_dir/key_pairs.list

if [ -f $key_pair_file ]; then
	rm -r $key_pair_file
fi

completed_jobs=`ls $key_pairs_dir/key_pairs.*.log | xargs grep "End time" | wc -l`
echo "Number of spawned jobs done: $completed_jobs out of $NUM_CORES"
if [ $NUM_CORES -ne $completed_jobs ]; then
	echo "Please wait for jobs to finish."
	exit 1
fi

if [ ! -f $key_pair_file ]; then
	touch $key_pair_file
	for i in $(seq 1 $NUM_CORES); do
		cat $key_pairs_dir/key_pairs.$i.list >> $key_pair_file
	done
fi
