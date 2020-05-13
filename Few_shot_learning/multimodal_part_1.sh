#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script calculates the distances between each spoken word and every other spoken word in 
# the dataset using DTW. 
#

set -e

NUM_CORES=6
cmd=./run.py
export PYTHONUNBUFFERED="YOUR_SET"

episode_file=$1
features_npz=$2
M=$3
K=$4
Q=$5
if [ -z $features_npz ]; then 
    echo "Feature file needed"
    exit 1
fi
if [ ! -f $features_npz ]; then
    echo "Feature file does not exist"
    exit 1
fi
   
name=`basename $episode_file`
name="${name%.*}"   
mm_dir=Experiments/Multimodal/$name
pair_ids=$mm_dir/speech_pairs.list
speech_pairs_dir=$mm_dir/speech_pair_lists
labels=$mm_dir/input_labels.list
episode_order=$mm_dir/episode_order.list
support_set_pairs=$mm_dir/support_set_pairs.list
speech_distance_dir=$mm_dir/speech_distance_lists
test_labels=$mm_dir/speech_labels.list

if [ -f $pair_ids ]; then
	rm -r $pair_ids
fi

if [ -d $speech_pairs_dir ]; then
	rm -r $speech_pairs_dir
fi

if [ -f $labels ]; then
	rm -r $labels
fi

if [ -f $episode_order ]; then
	rm -r $episode_order
fi

if [ -f $support_set_pairs ]; then
	rm -r $support_set_pairs
fi

if [ -d $speech_distance_dir ]; then
	rm -r $speech_distance_dir
fi

if [ -f $test_labels ]; then
	rm -r $test_labels
fi

[ ! -d $mm_dir ] && mkdir -p $mm_dir
[ ! -f $pair_ids ] && ./multimodal_speech_pairs.py $episode_file $pair_ids $labels $episode_order $support_set_pairs $NUM_CORES $speech_pairs_dir $M $K $test_labels

if [ ! -d $speech_distance_dir ]; then
	mkdir -p $speech_distance_dir
	dist_cmd="./dtw.py --pairs_fn $speech_pairs_dir/speech_pairs.JOB.list \
	--feats_fn $features_npz --distances_fn $speech_distance_dir/distances.JOB.dist \
	--binary_distances True --metric cosine"
	$cmd 1 $NUM_CORES $speech_distance_dir/distances.JOB.log "$dist_cmd"
fi

echo "Wait for jobs to finish before running multimodal_part_2.sh."