#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script calculates the distances between each image and every other image in the dataset. 
#

set -e

NUM_CORES=6
cmd=./run.py
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
pair_ids=$im_dir/pairs.list
pair_files_split_dir=$im_dir/pair_lists
labels=$im_dir/labels.list
distance_files_split_dir=$im_dir/distance_lists
distances=$im_dir/distances.dists

if [ -f $pair_ids ]; then
	rm -r $pair_ids
fi

if [ -d $pair_files_split_dir ]; then
	rm -r $pair_files_split_dir
fi

if [ -f $labels ]; then
	rm -r $labels
fi

if [ -d $distance_files_split_dir ]; then
	rm -r $distance_files_split_dir
fi

if [ -f $distances ]; then
	rm -r $distances
fi

[ ! -d $im_dir ] && mkdir -p $im_dir
[ ! -f $pair_ids ] && ./image_pairs.py $episode_file $pair_ids $labels $NUM_CORES $pair_files_split_dir $M $K
if [ ! -d $distance_files_split_dir ]; then
	mkdir -p $distance_files_split_dir
	dist_cmd="./distances.py --pairs_fn $pair_files_split_dir/pairs.JOB.list \
	--feats_fn $features_npz --distances_fn $distance_files_split_dir/distances.JOB.dist \
	--binary_distances True --metric cosine"
	$cmd 1 $NUM_CORES $distance_files_split_dir/distances.JOB.log "$dist_cmd"
fi

echo "Wait for jobs to finish before running unimodal_images_part_2.sh."