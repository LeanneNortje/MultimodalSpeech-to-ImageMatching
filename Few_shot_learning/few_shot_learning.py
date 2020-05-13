#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script completes the multimodal speech-image matching task by using the speech model given 
# by speech_lib_fn and the image model given by image_lib_fn.
#

from __future__ import division
from __future__ import print_function
from os import path
import argparse
import os
from datetime import datetime
import numpy as np
import sys
import time
import few_shot_learning_library
from scipy.spatial.distance import cdist
import timeit
from tqdm import tqdm
import re
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append("..")
from paths import feats_path
from paths import model_lib_path
from paths import general_lib_path
from paths import data_lib_path
from paths import few_shot_lib_path
from paths import results_path

sys.path.append(path.join("..", model_lib_path))
import model_setup_library
import model_legos_library
import speech_model_library
import vision_model_library

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", data_lib_path))
import data_library
import batching_library

sys.path.append(path.join("..", few_shot_lib_path))
import generate_episodes

feats_path = path.join("..", feats_path)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

PRINT_LENGTH = model_setup_library.PRINT_LENGTH
COL_LENGTH = model_setup_library.COL_LENGTH

#_____________________________________________________________________________________________________________________________________
#
# Default library
#
#_____________________________________________________________________________________________________________________________________

default_model_lib = {
        
        "max_frames": 101,
        "normalize": True,
        "final_model": True

    }

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________

def arguments_for_library_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_log_fn", type=str)
    parser.add_argument("--image_log_fn", type=str)
    parser.add_argument("--speech_data_fn", type=str)
    parser.add_argument("--image_data_fn", type=str)
    parser.add_argument("--episode_fn", type=str)
    parser.add_argument("--max_frames", type=int, default=default_model_lib["max_frames"])
    parser.add_argument("--normalize", type=str, choices=["True", "False"], default=str(default_model_lib["normalize"]))
    parser.add_argument("--final_model", type=str, choices=["True", "False"], default=str(default_model_lib["final_model"]))
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Library setup
#
#_____________________________________________________________________________________________________________________________________

def library_setup():

    parameters = arguments_for_library_setup()

    model_lib = default_model_lib.copy()

    model_lib["speech_log_fn"] = parameters.speech_log_fn
    model_lib["image_log_fn"] = parameters.image_log_fn
    model_lib["speech_data_fn"] = parameters.speech_data_fn
    model_lib["image_data_fn"] = parameters.image_data_fn
    model_lib["episode_fn"] = parameters.episode_fn
    model_lib["max_frames"] = parameters.max_frames 

    
    model_lib["speech_input_dim"] = 13
    model_lib["image_input_dim"] = 28*28
    model_lib["normalize"] = parameters.normalize = "True"
    model_lib["final_model"] = parameters.final_model == "True"

    return model_lib

#_____________________________________________________________________________________________________________________________________
#
# Retrieve embeddings for data inputs from specified model
#
#_____________________________________________________________________________________________________________________________________

def speech_rnn_latent_values(lib, iterator1, iterator2):
	
	tf.reset_default_graph()

	train_flag = tf.placeholder_with_default(False, shape=())

	if lib["model_type"] == "classifier":
		X = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
		target = tf.placeholder(tf.int32, [None, lib["num_classes"]])
		X_lengths = tf.placeholder(tf.int32, [None])
		train_flag = tf.placeholder_with_default(False, shape=())
		model = model_legos_library.rnn_classifier_architecture(
			[X, X_lengths], train_flag, model_setup_library.activation_lib(), lib, print_layer=False
			)
		output = model["output"]
		latent = model["latent"]
	elif lib["model_type"] == "siamese":
		X = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
		target = tf.placeholder(tf.int32, [None])
		X_lengths = tf.placeholder(tf.int32, [None])
		train_flag = tf.placeholder_with_default(False, shape=())
		model = model_legos_library.siamese_rnn_architecture(
			[X, X_lengths], train_flag, model_setup_library.activation_lib(), lib, print_layer=False
			)
		latent = tf.nn.l2_normalize(model["output"], axis=1)
		output = latent
	else:
		X = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
		target = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
		X_lengths = tf.placeholder(tf.int32, [None])
		target_lengths = tf.placeholder(tf.int32, [None])
		X_mask = tf.placeholder(tf.float32, [None, None])
		target_mask = tf.placeholder(tf.float32, [None, None])
		model = model_legos_library.rnn_architecture(
			[X, X_lengths], train_flag, model_setup_library.activation_lib(), 
			lib, target_lengths, print_layer=False
			)
		output = model["output"]
		latent = model["latent"]

	model_fn = model_setup_library.get_model_fn(lib)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	saver = tf.train.Saver()
	with tf.Session(config=config) as sesh:
		saver.restore(sesh, model_fn)
		for feats, lengths in iterator1:
			lat1 = sesh.run(
				[latent], feed_dict={X: feats, X_lengths: lengths, train_flag: False}
				)[0]

		for feats, lengths in iterator2:
			lat2 = sesh.run(
				[latent], feed_dict={X: feats, X_lengths: lengths, train_flag: False}
				)[0]
	return lat1, lat2


def image_fc_latent_values(lib, iterator1, iterator2):

	tf.reset_default_graph()

	X = tf.placeholder(tf.float32, [None, 28, 28, 1])
	train_flag = tf.placeholder_with_default(False, shape=())

	if lib["model_type"] == "classifier":
		target =  tf.placeholder(tf.float32, [None, lib["num_classes"]])
		model = model_legos_library.cnn_classifier_architecture(
			X, train_flag, lib["enc"], lib["enc_strides"], 
			model_setup_library.pooling_lib(), lib["pool_layers"], lib["latent"],
			lib, model_setup_library.activation_lib(), print_layer=False
			)

		output = model["output"]
		latent = model["latent"]
	elif lib["model_type"] == "siamese":
		target =  tf.placeholder(tf.float32, [None])
		model = model_legos_library.siamese_cnn_architecture(
			X, train_flag, lib["enc"], lib["enc_strides"], 
			model_setup_library.pooling_lib(), lib["pool_layers"], lib["latent"],
			lib, model_setup_library.activation_lib(), print_layer=False
			)

		latent = tf.nn.l2_normalize(model["output"], axis=1)
		output = latent
	else:
		target = tf.placeholder(tf.float32, [None, 28, 28, 1])
		model = model_legos_library.cnn_architecture(
			X, train_flag, lib["enc"], lib["enc_strides"], 
			model_setup_library.pooling_lib(), lib["pool_layers"], lib["latent"],
			lib["dec"], lib["dec_strides"], lib, model_setup_library.activation_lib(), print_layer=False
			)

		output = model["output"]
		latent = model["latent"]

	model_fn = model_setup_library.get_model_fn(lib)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	saver = tf.train.Saver()
	with tf.Session(config=config) as sesh:
		saver.restore(sesh, model_fn)

		for feats in iterator1:
			lat1 = sesh.run(
				[latent], feed_dict={X: feats, train_flag: False}
				)[0]

		for feats in iterator2:
			lat2 = sesh.run(
				[latent], feed_dict={X: feats, train_flag: False}
				)[0]

	return lat1, lat2

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

	lib = library_setup()

	print("\nTask information:")
	model_setup_library.lib_print(lib)

	# Load speech data
	print("\n" + "-"*PRINT_LENGTH)
	print("Speech data processing")
	print("-"*PRINT_LENGTH)
	sp_test_x, sp_test_labels, sp_test_lengths, sp_test_keys = (
		data_library.load_speech_data_from_npz(lib["speech_data_fn"])
		)
	max_frames = lib["max_frames"]
	d_frame = lib["speech_input_dim"]

	data_library.truncate_data_dim(sp_test_x, sp_test_lengths, d_frame, max_frames)

	# Load image data
	print("\n" + "-"*PRINT_LENGTH)
	print("Image data processing")
	print("-"*PRINT_LENGTH)
	im_test_x, im_test_labels, im_test_keys = (
		data_library.load_image_data_from_npz(lib["image_data_fn"])
		)

	# Load in episodes
	episode_dict = generate_episodes.read_in_episodes(
		lib["episode_fn"]
		)

	print("\n" + "-"*PRINT_LENGTH)
	print("Setting up seed pairs")
	print("-"*PRINT_LENGTH)

	path_parts = lib["speech_log_fn"].split("/")
	log_fn = lib["speech_log_fn"]
	speech_model_info = {}
	rnd_seeds = []
	for line in open(log_fn, 'r'):
		if re.search("rnd_seed", line):
			model_instance = line.split(": ")[0]
			line_parts = line.split()
			ind = np.where(np.asarray(line_parts) == "rnd_seed")[0][0]
			ind += 2
			this_rnd_seed = line_parts[ind]
			if this_rnd_seed not in rnd_seeds:
				temp_path_parts = path_parts.copy()
				temp_path_parts[-1] = model_instance
				temp_path = "/".join(temp_path_parts)
				speech_model_info[this_rnd_seed] = path.join(temp_path, temp_path_parts[-2] + "_lib.pkl")

	path_parts = lib["image_log_fn"].split("/")
	log_fn = lib["image_log_fn"]
	image_model_info = {}
	rnd_seeds = []
	for line in open(log_fn, 'r'):
		if re.search("rnd_seed", line):
			model_instance = line.split(": ")[0]
			line_parts = line.split()
			ind = np.where(np.asarray(line_parts) == "rnd_seed")[0][0]
			ind += 2
			this_rnd_seed = line_parts[ind]
			if this_rnd_seed not in rnd_seeds:
				temp_path_parts = path_parts.copy()
				temp_path_parts[-1] = model_instance
				temp_path = "/".join(temp_path_parts)
				image_model_info[this_rnd_seed] = path.join(temp_path, temp_path_parts[-2] + "_lib.pkl")

	model_info = []
	for key in speech_model_info:
		model_info.append((int(key), speech_model_info[key], image_model_info[key]))

	for (rnd_seed, speech_lib_fn, image_lib_fn) in model_info:
		np.random.seed(rnd_seed)
		episode_numbers = np.arange(1, len(episode_dict)+1)
		np.random.shuffle(episode_numbers)
		correct = 0
		total = 0

		speech_lib = model_setup_library.restore_lib(speech_lib_fn)
		image_lib = model_setup_library.restore_lib(image_lib_fn)

		results_dir = path.join("..", results_path, "Multimodal_results")

		episode_list_parts = lib["episode_fn"].split("_")
		K = episode_list_parts[np.where(np.asarray(episode_list_parts) == "K")[0][0] + 1] 

		model_base = "speech_{}_image_{}".format(speech_lib["model_name"], image_lib["model_name"])
		model_name = "{}_shot".format(K)
		results_dir = path.join(results_dir, model_base)
		util_library.check_dir(results_dir)
		results_fn = path.join(results_dir, model_name + "_results.txt")

		keyword = "{}-shot accuracy of ".format(K)
		record_dict = {}
		if os.path.isfile(results_fn) and path.isdir(results_dir):

			for line in open(results_fn, 'r'):

				if re.search(keyword, line):
					line_parts = line.strip().split(" ")
					keyword_parts = keyword.strip().split(" ")
					ind = np.where(np.asarray(line_parts) == keyword_parts[0])[0][0]
					acc = float(line_parts[ind+3])
					old_rnd_seed = int(line_parts[-1])

					if old_rnd_seed not in record_dict:
						record_dict[old_rnd_seed] = acc

		if rnd_seed not in record_dict:

			for episode in tqdm(episode_numbers, desc=f'\tMultimodal tests on {len(episode_numbers)} episodes for random seed {rnd_seed:2.0f}', ncols=COL_LENGTH):

				episode_num = str(episode)

				# Get query iterator
				query = episode_dict[episode_num]["query"]
				query_data, query_keys, query_lab = generate_episodes.episode_data(
					query["keys"], sp_test_x, sp_test_keys, sp_test_labels
					)
				query_iterator = batching_library.speech_iterator(
		            query_data, len(query_data), shuffle_batches_every_epoch=False
		            )

				# Get speech_support set
				support_set = episode_dict[episode_num]["support_set"]
				S_image_data, S_image_keys, S_image_lab = generate_episodes.episode_data(
					support_set["image_keys"], im_test_x, im_test_keys, im_test_labels
					)
				S_speech_data, S_speech_keys, S_speech_lab = generate_episodes.episode_data(
					support_set["speech_keys"], sp_test_x, sp_test_keys, sp_test_labels
					)
				key_list = []
				for i in range(len(S_speech_keys)):
					key_list.append((S_speech_keys[i], S_image_keys[i]))

				S_speech_iterator = batching_library.speech_iterator(
		            S_speech_data, len(S_speech_data), shuffle_batches_every_epoch=False
		            )
				
				query_latents, s_speech_latents = speech_rnn_latent_values(speech_lib, query_iterator, S_speech_iterator)
				query_latent_labels = [query_lab[i] for i in query_iterator.indices]
				query_latent_keys = [query_keys[i] for i in query_iterator.indices]
				s_speech_latent_labels = [S_speech_lab[i] for i in S_speech_iterator.indices]
				s_speech_latent_keys = [S_speech_keys[i] for i in S_speech_iterator.indices]

				if lib["normalize"]:
					query_latents = (query_latents - query_latents.mean(axis=0))/query_latents.std(axis=0)
					s_speech_latents = (s_speech_latents - s_speech_latents.mean(axis=0))/s_speech_latents.std(axis=0)

				distances1 = cdist(query_latents, s_speech_latents, "cosine")
				indexes1 = np.argmin(distances1, axis=1)
			

				chosen_speech_keys = []
				for i in range(len(indexes1)):
					chosen_speech_keys.append(s_speech_latent_keys[indexes1[i]])

				S_image_iterator = batching_library.unflattened_image_iterator(
		            S_image_data, len(S_image_data), shuffle_batches_every_epoch=False
		            )
				matching_set = episode_dict[episode_num]["matching_set"]
				M_data, M_keys, M_lab = generate_episodes.episode_data(
					matching_set["keys"], im_test_x, im_test_keys, im_test_labels
					)
				M_image_iterator = batching_library.unflattened_image_iterator(
		            M_data, len(M_data), shuffle_batches_every_epoch=False
		            )

				s_image_latents, matching_latents = image_fc_latent_values(image_lib, S_image_iterator, M_image_iterator)
				s_image_latent_labels = [S_image_lab[i] for i in S_image_iterator.indices]
				s_image_latent_keys = [S_image_keys[i] for i in S_image_iterator.indices]

				matching_latent_labels = [M_lab[i] for i in M_image_iterator.indices]
				matching_latent_keys = [M_keys[i] for i in M_image_iterator.indices]

				image_key_order_list = [] #just a check remove later
				s_image_latents_in_order = np.empty((query_latents.shape[0], s_image_latents.shape[1]))
				s_image_labels_in_order = [] #just a check remove later

				for j, key in enumerate(chosen_speech_keys):

					for (sp_key, im_key) in key_list:

						if key == sp_key:
							image_key_order_list.append(im_key)
							i = np.where(np.asarray(s_image_latent_keys) == im_key)[0][0]
							s_image_latents_in_order[j:j+1, :] = s_image_latents[i:i+1, :]
							s_image_labels_in_order.append(s_image_latent_labels[i])
							break

				if lib["normalize"]:
					s_image_latents_in_order = (s_image_latents_in_order - s_image_latents_in_order.mean(axis=0))/s_image_latents_in_order.std(axis=0)
					matching_latents = (matching_latents - matching_latents.mean(axis=0))/matching_latents.std(axis=0)

				distances2 = cdist(s_image_latents_in_order, matching_latents, "cosine")
				indexes2 = np.argmin(distances2, axis=1)
				label_matches = few_shot_learning_library.label_matches_grid_generation_2D(query_latent_labels, matching_latent_labels)

				for i in range(len(indexes2)):
					total += 1
					if label_matches[i, indexes2[i]]:
						correct += 1

			acc = correct/total
			print(f'\tAccuracy: {acc*100:3.2f}%')

			if os.path.isfile(results_fn) and path.isdir(results_dir):
				with open(results_fn, 'a') as results_file:
					if K == 1: results_file.write("One-shot accuracy of {} at rnd_seed of {} \n".format(acc, rnd_seed))
					else: results_file.write("{}-shot accuracy of {} at rnd_seed of {} \n".format(K, acc, rnd_seed)) 
			else:
				with open(results_fn, 'a') as results_file:
					date = str(datetime.now()).split(" ")
					results_file.write("Model name: {}\nLog file was created on {} at {}\n\n".format(model_name, date[0], date[1]))
					if K == 1: results_file.write("One-shot accuracy of {} at rnd_seed of {} \n".format(acc, rnd_seed))
					else: results_file.write("{}-shot accuracy of {} at rnd_seed of {} \n".format(K, acc, rnd_seed)) 
			results_file.close()
			run_flag = 1
		else:
			print(f'Test for rnd_seed {rnd_seed:>3} already done.')
			run_flag = 0

		print(f'\tTask progress info:')
		print(f'\t{len(speech_model_info)} speech model instances found.')
		print(f'\t{len(image_model_info)} image model instances found.')
		print(f'\t{len(model_info)} model instance pairs found.')
		print(f'\t{len(record_dict)+run_flag} model instance pairs already tested on the multimodal task.\n')


if __name__ == "__main__":
	main()