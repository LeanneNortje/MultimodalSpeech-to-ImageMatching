#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script identifies the closest spoken word in the support set to the query spoken word. It 
# then takes the closest spoken words' paired image and calculates the distances between this 
# paired image and eeach image in the matching set.  
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
speech_distance_dir=$mm_dir/speech_distance_lists
speech_distances=$mm_dir/speech_distances.dists
pair_ids=$mm_dir/image_pairs.list
episode_order=$mm_dir/episode_order.list
labels=$mm_dir/input_labels.list
out_labels=$mm_dir/output_labels.list
speech_keys=$mm_dir/speech_pairs.list
image_keys=$mm_dir/image_pairs.list
image_pair_dir=$mm_dir/image_pair_lists
image_distance_dir=$mm_dir/image_distance_lists
support_set_pairs=$mm_dir/support_set_pairs.list
test_fn=$mm_dir/test.txt
test_labels=$mm_dir/speech_labels.list

if [ -f $pair_ids ]; then
	rm -r $pair_ids
fi

if [ -f $out_labels ]; then
	rm -r $out_labels
fi

if [ -f $image_keys ]; then
	rm -r $image_keys
fi

if [ -d $image_pair_dir ]; then
	rm -r $image_pair_dir
fi

if [ -d $image_distance_dir ]; then
	rm -r $image_distance_dir
fi


completed_jobs=`ls $speech_distance_dir/distances.*.log | xargs grep "End time" | wc -l`
echo "Number of spawned jobs done: $completed_jobs out of $NUM_CORES"
if [ $NUM_CORES -ne $completed_jobs ]; then
	echo "Please wait for jobs to finish."
	exit 1
fi

if [ ! -f $speech_distances ]; then
	touch $speech_distances
	for i in $(seq 1 $NUM_CORES); do
		cat $speech_distance_dir/distances.$i.dist >> $speech_distances
	done
fi

[ ! -f $pair_ids ] && ./multimodal_image_pairs.py --episodes_fn $episode_file --distances_fn $speech_distances \
	--episode_order_fn $episode_order --support_set_fn $support_set_pairs --labels_fn $labels \
	--pair_labels_fn $test_labels --output_labels_fn $out_labels --keys_fn $speech_keys --binary_dists True \
	--m_way $M --k_shot $K --output_keys_fn $image_keys --num_files $NUM_CORES --output_dir $image_pair_dir \
	--test_fn $test_fn
if [ ! -d $image_distance_dir ]; then
	mkdir -p $image_distance_dir
	dist_cmd="./distances.py --pairs_fn $image_pair_dir/image_pairs.JOB.list \
	--feats_fn $features_npz --distances_fn $image_distance_dir/distances.JOB.dist \
	--binary_distances True --metric cosine"
	$cmd 1 $NUM_CORES $image_distance_dir/distances.JOB.log "$dist_cmd"
fi

echo "Wait for jobs to finish before running multimodal_part_3.sh."