#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script identifies the closest image in the matching set to the paired image in the support 
# set and averages this over the N number of episodes. 
#

NUM_CORES=6
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
image_distance_dir=$mm_dir/image_distance_lists
image_distances=$mm_dir/image_distances.dists
out_labels=$mm_dir/output_labels.list
one_shot_result=$mm_dir/results.txt
test_fn=$mm_dir/test.list

if [ -f $one_shot_result ]; then
	rm -r $one_shot_result
fi

completed_jobs=`ls $image_distance_dir/distances.*.log | xargs grep "End time" | wc -l`
echo "Number of spawned jobs done: $completed_jobs out of $NUM_CORES"
if [ $NUM_CORES -ne $completed_jobs ]; then
	echo "Please wait for jobs to finish."
	exit 1
fi

if [ ! -f $image_distances ]; then
	touch $image_distances
	for i in $(seq 1 $NUM_CORES); do
		cat $image_distance_dir/distances.$i.dist >> $image_distances
	done
fi

if [ ! -f $one_shot_result ]; then
	./few_shot_task.py --distances_fn $image_distances --labels_fn $out_labels --binary_dists True \
		--m_way $Q --k_shot 1 --test_fn $test_fn > $one_shot_result
	cat $one_shot_result

	if [ $? -ne 0 ]; then
		echo "Oops, something went wrong. Exiting now and deleting $one_shot_result."
		rm $one_shot_result
		exit 1
	fi
fi

echo "Done :)"