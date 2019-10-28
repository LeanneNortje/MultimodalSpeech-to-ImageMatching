#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script uses the distances between each image and every other image in the dataset 
# calculated in unimodal_images_part_1.sh to complerte the unimodal image classification task. 
#

NUM_CORES=6
export PYTHONUNBUFFERED="YOUR_SET"

episode_file=$1
features_npz=$2
M=$3
K=$4
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
im_dir=Experiments/Unimodal_image_experiments/$name
labels=$im_dir/labels.list
distance_files_split_dir=$im_dir/distance_lists
distances=$im_dir/distances.dists
one_shot_result=$im_dir/result.list
test_fn=$dtw_dir/test.list

if [ -f $one_shot_result ]; then
	rm -r $one_shot_result
fi

completed_jobs=`ls $distance_files_split_dir/distances.*.log | xargs grep "End time" | wc -l`
echo "Number of spawned jobs done: $completed_jobs out of $NUM_CORES"
if [ $NUM_CORES -ne $completed_jobs ]; then
	echo "Please wait for jobs to finish."
	exit 1
fi

if [ ! -f $distances ]; then
	touch $distances
	for i in $(seq 1 $NUM_CORES); do
		cat $distance_files_split_dir/distances.$i.dist >> $distances
	done
fi

if [ ! -f $one_shot_result ]; then
	./few_shot_task.py --distances_fn $distances --labels_fn $labels --binary_dists True \
		--m_way $M --k_shot $K --test_fn $test_fn > $one_shot_result
	cat $one_shot_result

	if [ $? -ne 0 ]; then
		echo "Oops, something went wrong. Exiting now and deleting $one_shot_result."
		rm $one_shot_result
		exit 1
	fi
fi

echo "Done :)"