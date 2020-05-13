#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script generates unimodal image episodes. 
#

from __future__ import division
from __future__ import print_function
from os import path
import argparse
import os
import datetime
import numpy as np
import sys
import time
from scipy.spatial.distance import cdist
import hashlib
import few_shot_learning_library
import re
import filecmp
from tqdm import tqdm

sys.path.append("..")
from paths import data_path
from paths import feats_path
from paths import data_lib_path
from paths import general_lib_path
from paths import model_lib_path

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", data_lib_path))
import data_library
import batching_library

sys.path.append(path.join("..", model_lib_path))
import model_setup_library
import speech_model_library
import vision_model_library

data_path = path.join("..", data_path)
feats_path = path.join("..", feats_path)

import random

#_____________________________________________________________________________________________________________________________________
#
# Main 
#
#_____________________________________________________________________________________________________________________________________

default_model_lib = {
    
        "data_type": "MNIST",
        "subset": "test",        

    }

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def arguments_for_library_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, choices=["MNIST", "omniglot"], default=default_model_lib["data_type"])
    parser.add_argument("--subset", type=str, choices=["train", "val", "test"], default=default_model_lib["subset"])
    parser.add_argument("--M", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--Q", type=int)    
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Library setup
#
#_____________________________________________________________________________________________________________________________________

def library_setup():

    parameters = arguments_for_library_setup()

    model_lib = default_model_lib.copy()

    model_lib["data_type"] = parameters.data_type
    model_lib["subset"] = parameters.subset
    model_lib["M"] = parameters.M
    model_lib["K"] = parameters.K
    model_lib["Q"] = parameters.Q

    if model_lib["subset"] == "val": 
    	subset = "validation.npz"
    else: 
    	subset = model_lib["subset"] + ".npz" 

    model_lib["data_dir"] = path.join(
    	feats_path, model_lib["data_type"], subset
    	)

    model_lib["image_input_dim"] = 28*28

    model_lib["name"] = "M_{}_K_{}_Q_{}_{}_{}".format(
    		model_lib["M"], model_lib["K"], model_lib["Q"], model_lib["data_type"],
    		model_lib["subset"]
    		)

    base_dir = path.join(".", "Episode_files")
    util_library.check_dir(base_dir)
    model_lib["data_fn"] = path.join(base_dir, model_lib["name"])

    return model_lib

#_____________________________________________________________________________________________________________________________________
#
# Retrieving data of certain keys from dataset
#
#_____________________________________________________________________________________________________________________________________

def episode_data(query_keys, data, keys, labels):

	data_list = []
	key_list = []
	label_list = []

	for i in range(len(query_keys)):

		index = np.where(np.asarray(keys) == query_keys[i])[0][0]
		if len(data[index].shape) == 1: data_list.append(data[index].reshape((1, data[index].shape[-1])))
		else: data_list.append(data[index])
		key_list.append(keys[index])
		label_list.append(labels[index])

	return data_list, key_list, label_list

#_____________________________________________________________________________________________________________________________________
#
# Ensuring none of the MNIST classes occur in the Omniglot dataset
#
#_____________________________________________________________________________________________________________________________________

def filter_omniglot_set(subset, subset_keys, subset_labels, M, K, Q):

	minimum = K + 2

	label_count = {}

	for label in subset_labels:
		if label not in label_count:
			label_count[label] = 1

		else:
			label_count[label] += 1

	valid_labels = []

	for label in label_count:
		if label_count[label] >= minimum: valid_labels.append(label)

	updated_subset = []
	updated_labels = []
	updated_keys = []

	for i in range(len(subset_labels)):
		if subset_labels[i] in valid_labels:
			updated_subset.append(subset[i])
			updated_keys.append(subset_keys[i])
			updated_labels.append(subset_labels[i])

	return updated_subset, updated_keys, updated_labels

#_____________________________________________________________________________________________________________________________________
#
# Readin in the keys in each set that makes up an episode 
#
#_____________________________________________________________________________________________________________________________________

def read_in_episodes(list_fn):

	index_dict = {}
	curr_episode = ""
	episodes = {}
	currently_reading_in = ""
	currently_reading_in_section = "" 

	for i, line in enumerate(open(list_fn, 'r')):
		if re.search("Episode", line): 
			currently_reading_in = ""
			episode_num = line.strip().split(" ")[-1]
			curr_episode = str(episode_num)
			episodes[curr_episode] = {}

		elif re.search("Support set:", line): 
			currently_reading_in = "support_set"
			episodes[curr_episode]["support_set"] = {}
			episodes[curr_episode]["support_set"]["labels"] = []
			episodes[curr_episode]["support_set"]["keys"] = []
		elif re.search("Query:", line): 
			currently_reading_in = "query"
			episodes[curr_episode]["query"] = {}
			episodes[curr_episode]["query"]["labels"] = []
			episodes[curr_episode]["query"]["keys"] = []			
		elif re.search("Matching set:", line): 
			currently_reading_in = "matching_set"
			episodes[curr_episode]["matching_set"] = {}
			episodes[curr_episode]["matching_set"]["labels"] = []
			episodes[curr_episode]["matching_set"]["keys"] = []

		elif re.search("Labels:", line):
			currently_reading_in_section = "labels"
		elif re.search("Keys:", line):
			currently_reading_in_section = "keys"
		elif len(line.strip().split()) == 0:
			currently_reading_in = ""
			currently_reading_in_section = ""
			continue

		elif currently_reading_in == "support_set" and currently_reading_in_section == "labels":
			line_parts = line.strip().split()
			episodes[curr_episode]["support_set"]["labels"].append(line_parts[0])
		
		elif currently_reading_in == "support_set" and currently_reading_in_section == "keys":
			line_parts = line.strip().split()
			key = []
			for i in range(len(line_parts)): key.append(line_parts[i])
			episodes[curr_episode]["support_set"]["keys"].extend(key)

		elif currently_reading_in == "query" and currently_reading_in_section == "labels":
			line_parts = line.strip().split()
			episodes[curr_episode]["query"]["labels"].append(line_parts[0])
		elif currently_reading_in == "query" and currently_reading_in_section == "keys":
			line_parts = line.strip().split()
			key = []
			for i in range(len(line_parts)): key.append(line_parts[i])
			episodes[curr_episode]["query"]["keys"].extend(key)

		elif currently_reading_in == "matching_set" and currently_reading_in_section == "labels":
			line_parts = line.strip().split()
			episodes[curr_episode]["matching_set"]["labels"].append(line_parts[0])
		elif currently_reading_in == "matching_set" and currently_reading_in_section == "keys":
			line_parts = line.strip().split()
			key = []
			for i in range(len(line_parts)): key.append(line_parts[i])
			episodes[curr_episode]["matching_set"]["keys"].extend(key)

	return episodes

#_____________________________________________________________________________________________________________________________________
#
# Checking that keys of the queries, support set and matching set do not overlap
#
#_____________________________________________________________________________________________________________________________________

def episode_check(support_set, query, matching_set):
	check = False
	total = 0
	correct = 0
	im_keys = []
	speech_keys = []
	support_set_image_keys = []
	support_set_speech_keys = []
	query_speech_keys = []
	matching_set_image_keys = []

	for key in support_set:
		support_set_speech_keys.extend(support_set[key]["image_keys"])

	for key in query:
		query_speech_keys.extend(query[key]["keys"])

	for key in matching_set:
		matching_set_image_keys.extend(matching_set[key]["keys"])

	for support_key in support_set_image_keys:
		if support_key not in matching_set_image_keys:
			correct += 1
		else:
			im_keys.append(support_key)
		total += 1

	for support_key in support_set_speech_keys:
		if support_key not in query_speech_keys: 
			correct += 1
		else:
			im_keys.append(support_key)
		total += 1

	if correct != total: 
		check = True

	return check
	
#_____________________________________________________________________________________________________________________________________
#
# Main 
#
#_____________________________________________________________________________________________________________________________________

def main():

	lib = library_setup()

	num_episodes = 400
	K = lib["K"]
	M = lib["M"]
	Q = lib["Q"]

	images, image_labels, image_keys = (
		data_library.load_image_data_from_npz(lib["data_dir"])
		)
	im_x, im_labels, im_keys  = images, image_labels, image_keys

	if lib["data_type"] == "omniglot":
		images, image_keys, image_labels = filter_omniglot_set(images, image_keys, image_labels, M, K, Q)
	
	episodes_fn = lib["data_fn"] + "_episodes.txt"
	print("Generating epsiodes...\n")
	
	file = open(episodes_fn, "w")
	
	for episode_counter in tqdm(range(1, num_episodes+1)):
	
		support_set = few_shot_learning_library.construct_few_shot_support_set_with_keys(
			sp_x=None, sp_labels=None, sp_keys=None, sp_lengths=None, 
			im_x=images, im_labels=image_labels, im_keys=image_keys, num_to_sample=M, num_of_each_sample=K, 
			do_switch=False
			)

		im_keys_to_not_include = []
		labels_wanted = []
		for key in support_set:
			labels_wanted.append(key)
			for im_key in support_set[key]["image_keys"]: im_keys_to_not_include.append(im_key)

		query_dict = few_shot_learning_library.sample_multiple_keys(
			images, image_labels, image_keys, num_to_sample=Q, 
			exclude_key_list=im_keys_to_not_include, labels_wanted=labels_wanted
			)	

		for key in query_dict:
			for im_key in query_dict[key]["keys"]: im_keys_to_not_include.append(im_key)


		matching_dict = few_shot_learning_library.sample_multiple_keys(
			images, image_labels, image_keys, num_to_sample=len(labels_wanted), num_of_each_sample=1, 
			exclude_key_list=im_keys_to_not_include, labels_wanted=labels_wanted
			)

		file.write("Episode {}\n".format(episode_counter))

		file.write("{}\n".format("Support set:"))
		file.write("{}\n".format("Labels: "))
		for key in support_set:
			file.write("{}\n".format(key))

		file.write("{}\n".format("Keys: "))
		for key in support_set:
			for i in range(len(support_set[key]["image_keys"])):
				file.write("{}".format(support_set[key]["image_keys"][i]))
				if i == len(support_set[key]["image_keys"]) - 1: file.write("\n")
				else: file.write(" ")

		file.write("{}\n".format("Query:"))
		file.write("{}\n".format("Labels: "))
		for key in query_dict:
			file.write("{}\n".format(key))
		file.write("{}\n".format("Keys: "))
		for key in query_dict:
			for i in range(len(query_dict[key]["keys"])):
				file.write("{}".format(query_dict[key]["keys"][i]))
				if i == len(query_dict[key]["keys"]) - 1: file.write("\n")
				else: file.write(" ")

		file.write("{}\n".format("Matching set:"))
		file.write("{}\n".format("Labels: "))
		for key in matching_dict:
			file.write("{}\n".format(key))
		file.write("{}\n".format("Keys: "))
		for key in matching_dict:
			for i in range(len(matching_dict[key]["keys"])):
				file.write("{}".format(matching_dict[key]["keys"][i]))
				if i == len(matching_dict[key]["keys"]) - 1: file.write("\n")
				else: file.write(" ")

		file.write("\n")

		if episode_check(support_set, query_dict, matching_dict):
			break

	file.close()

	print("Wrote epsiodes to {}".format(episodes_fn))

if __name__ == "__main__":
	main()