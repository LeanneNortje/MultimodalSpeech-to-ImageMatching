#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script trains and tests the different speech models.
#

from datetime import datetime
from os import path
import argparse
import glob
import numpy as np
import os
from os import path
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
import logging
import tensorflow as tf
import hashlib
import timeit
import sys
import subprocess
import pickle
from tqdm import tqdm
import random

import model_legos_library
import model_setup_library
import training_library

sys.path.append("..")
from paths import data_lib_path
from paths import evaluation_lib_path
from paths import general_lib_path
from paths import data_path
from paths import few_shot_lib_path
data_path = path.join("..", data_path)

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", data_lib_path))
import data_library
import batching_library

sys.path.append(path.join("..", few_shot_lib_path))
import few_shot_learning_library
import generate_unimodal_speech_episodes

PRINT_LENGTH = model_setup_library.PRINT_LENGTH

#__________________________________________________________________________________________________
#
# Model 
#
#__________________________________________________________________________________________________

def rnn_speech_model(lib):

    #______________________________________________________________________________________________
    # Model setup
    #______________________________________________________________________________________________

    np.random.seed(lib["rnd_seed"])
    tf.set_random_seed(lib["rnd_seed"])

    epochs = lib["epochs"]
    batch_size = lib["batch_size"] 
        
    tf.reset_default_graph()

    model_setup_library.lib_print(lib)

    #______________________________________________________________________________________________
    # Data processing
    #______________________________________________________________________________________________
    
    max_frames = lib["max_frames"]
    d_frame = lib["input_dim"]

    if lib["train_model"]:
        print("\n" + "-"*PRINT_LENGTH)
        print("Processing training data")
        print("-"*PRINT_LENGTH)
        train_x, train_labels, train_lengths, train_keys = (
        	data_library.load_speech_data_from_npz(lib["train_data_dir"])
        	)
        data_library.truncate_data_dim(train_x, train_lengths, d_frame, max_frames)
        train_mask = data_library.get_mask(
        	train_x, train_lengths, max_frames if lib["mix_training_datasets"] else None
        	)
        if lib["data_type"] == "buckeye":
            train_x, train_labels, train_lengths, train_keys = (
                data_library.remove_test_classes(
                    train_x, train_labels, train_lengths, train_keys, model_setup_library.DIGIT_LIST
                    )
                )
            data_library.test_classes(train_labels, model_setup_library.DIGIT_LIST, "training")

        if lib["pretrain"]:
            if lib["pretrain_train_data_dir"] == lib["train_data_dir"]:
                pretrain_train_x = train_x
                pretrain_train_labels = train_labels
                pretrain_train_lengths = train_lengths
                pretrain_train_keys = train_keys
                pretrain_train_mask = train_mask
            else:
                pretrain_train_x, pretrain_train_labels, pretrain_train_lengths, pretrain_train_keys = (
                    data_library.load_speech_data_from_npz(lib["pretrain_train_data_dir"])
                    )
                if lib["pretraining_data"] == "buckeye":
                    pretrain_train_x, pretrain_train_labels, pretrain_train_lengths, pretrain_train_keys = (
                        data_library.remove_test_classes(
                            pretrain_train_x, pretrain_train_labels, pretrain_train_lengths,
                            pretrain_train_keys, model_setup_library.DIGIT_LIST
                            )
                        )
                    data_library.test_classes(pretrain_train_labels, model_setup_library.DIGIT_LIST, "pretraining")

                data_library.truncate_data_dim(pretrain_train_x, pretrain_train_lengths, d_frame, max_frames)
                pretrain_train_mask = data_library.get_mask(
                    pretrain_train_x, pretrain_train_lengths, max_frames if lib["mix_training_datasets"] else None
                    )

        if (lib["mix_training_datasets"] and lib["data_type"] != lib["other_speech_dataset"]):
            if lib["other_train_data_dir"] == lib["pretrain_train_data_dir"] and lib["pretrain"]:
                other_train_x = pretrain_train_x
                other_train_labels = pretrain_train_labels
                other_train_lengths = pretrain_train_lengths
                other_train_keys = pretrain_train_keys
                other_train_mask = pretrain_train_mask
            else:
                other_train_x, other_train_labels, other_train_lengths, other_train_keys = (
                    data_library.load_speech_data_from_npz(lib["other_train_data_dir"])
                    )
                if lib["other_speech_dataset"] == "buckeye":
                    other_train_x, other_train_labels, other_train_lengths, other_train_keys = (
                        data_library.remove_test_classes(
                            other_train_x, other_train_labels, other_train_lengths, other_train_keys,
                            model_setup_library.DIGIT_LIST
                            )
                        )
                    data_library.test_classes(other_train_labels, model_setup_library.DIGIT_LIST, "other training")

                data_library.truncate_data_dim(other_train_x, other_train_lengths, d_frame, max_frames)
                other_train_mask = data_library.get_mask(other_train_x, other_train_lengths, max_frames)

            if (lib["pretrain"] and lib["pretraining_data"] != lib["other_pretraining_speech_dataset"]):
                if lib["other_pretrain_train_data_dir"] == lib["train_data_dir"]:
                    other_pretrain_train_x = train_x
                    other_pretrain_train_labels = train_labels
                    other_pretrain_train_lengths = train_lengths
                    other_pretrain_train_keys = train_keys
                    other_pretrain_train_mask = train_mask
                elif lib["other_pretrain_train_data_dir"] == lib["other_train_data_dir"]:
                    other_pretrain_train_x = other_train_x
                    other_pretrain_train_labels = other_train_labels
                    other_pretrain_train_lengths = other_train_lengths
                    other_pretrain_train_keys = other_train_keys
                    other_pretrain_train_mask = other_train_mask
                else:
                    other_pretrain_train_x, other_pretrain_train_labels, other_pretrain_train_lengths, other_pretrain_train_keys = (
                        data_library.load_speech_data_from_npz(lib["other_pretrain_train_data_dir"])
                        )
                    if lib["other_pretraining_speech_dataset"] == "buckeye":
                        other_pretrain_train_x, other_pretrain_train_labels, other_pretrain_train_lengths, other_pretrain_train_keys = (
                            data_library.remove_test_classes(
                                other_pretrain_train_x, other_pretrain_train_labels, other_pretrain_train_lengths,
                                other_pretrain_train_keys, model_setup_library.DIGIT_LIST
                                )
                            )
                        data_library.test_classes(other_pretrain_train_labels, model_setup_library.DIGIT_LIST, "other_pretraining")

                    data_library.truncate_data_dim(other_pretrain_train_x, other_pretrain_train_lengths, d_frame, max_frames)
                    other_pretrain_train_mask = data_library.get_mask(other_pretrain_train_x, other_pretrain_train_lengths, max_frames)

        print("\n" + "-"*PRINT_LENGTH)
        print("Processing validation data")
        print("-"*PRINT_LENGTH)
        val_x, val_labels, val_lengths, val_keys = (
            data_library.load_speech_data_from_npz(lib["val_data_dir"])
            )
        if (lib["validate_on_validation_dataset"] and lib["validation_speech_dataset"] == "buckeye") or (lib["validate_on_validation_dataset"] is False and lib["data_type"] == "buckeye"):
            val_x, val_labels, val_lengths, val_keys = (
                data_library.remove_test_classes(
                    val_x, val_labels, val_lengths, val_keys, model_setup_library.DIGIT_LIST
                    )
                )
            data_library.test_classes(val_labels, model_setup_library.DIGIT_LIST, "valdiation")

        data_library.truncate_data_dim(val_x, val_lengths, d_frame, max_frames)
        val_mask = data_library.get_mask(val_x, val_lengths)


    print("\n" + "-"*PRINT_LENGTH)
    print("Processing testing data")
    print("-"*PRINT_LENGTH)
    test_x, test_labels, test_lengths, test_keys = (
        data_library.load_speech_data_from_npz(lib["test_data_dir"])
        )
    if (lib["test_on_one_shot_dataset"] and lib["one_shot_speech_dataset"] == "buckeye") or (lib["test_on_one_shot_dataset"] is False and lib["data_type"] == "buckeye"):
        test_x, test_labels, test_lengths, test_keys = (
            data_library.remove_test_classes(
                test_x, test_labels, test_lengths, test_keys, model_setup_library.DIGIT_LIST
                )
            )
        data_library.test_classes(test_labels, model_setup_library.DIGIT_LIST, "testing")

    data_library.truncate_data_dim(test_x, test_lengths, d_frame, max_frames)
    test_mask = data_library.get_mask(test_x, test_lengths)

    if lib["train_model"] is False:
        val_x = test_x
        val_labels = test_labels
        val_lengths = test_lengths
        val_keys = test_keys

    #______________________________________________________________________________________________
    #Building model
    #______________________________________________________________________________________________

    print("\n" + "-"*PRINT_LENGTH)
    print("RNN structure setup")
    print("-"*PRINT_LENGTH)

    X = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
    target = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
    X_lengths = tf.placeholder(tf.int32, [None])
    target_lengths = tf.placeholder(tf.int32, [None])
    X_mask = tf.placeholder(tf.float32, [None, None])
    target_mask = tf.placeholder(tf.float32, [None, None])
    train_flag = tf.placeholder_with_default(False, shape=())
    training_placeholders = [X, X_lengths, X_mask, target, target_lengths, target_mask]
    
    model = model_legos_library.rnn_architecture(
        [X, X_lengths], train_flag, model_setup_library.activation_lib(), lib, target_lengths
        )

    output = model["output"]
    latent = model["latent"]

    masked_output = output * tf.expand_dims(target_mask, -1)

    loss = tf.reduce_mean(
        tf.reduce_sum(tf.reduce_mean(tf.square(target-masked_output), -1), -1) 
        / tf.reduce_sum(target_mask, 1)
        )

    optimization = tf.train.AdamOptimizer(lib["learning_rate"]).minimize(loss)

    #______________________________________________________________________________________________
    # One-shot evaluation
    #______________________________________________________________________________________________

    def one_shot_validation(episode_file=lib["validation_episode_list"], data_x=val_x, data_keys=val_keys, data_labels=val_labels, normalize=True, print_normalization=False):

        episode_dict = generate_unimodal_speech_episodes.read_in_episodes(episode_file)
        correct = 0
        total = 0

        episode_numbers = np.arange(1, len(episode_dict)+1)
        np.random.shuffle(episode_numbers)

        saver = tf.train.Saver()
        with tf.Session() as sesh:
            saver.restore(sesh, model_fn)

            for episode in episode_numbers:

                episode_num = str(episode)
                query = episode_dict[episode_num]["query"]
                query_data, query_keys, query_lab = generate_unimodal_speech_episodes.episode_data(
                    query["keys"], data_x, data_keys, data_labels
                    )
                query_iterator = batching_library.speech_iterator(
                    query_data, len(query_data), shuffle_batches_every_epoch=False
                    )
                query_labels = [query_lab[i] for i in query_iterator.indices]

                support_set = episode_dict[episode_num]["support_set"]
                S_data, S_keys, S_lab = generate_unimodal_speech_episodes.episode_data(
                    support_set["keys"], data_x, data_keys, data_labels
                    )
                S_iterator = batching_library.speech_iterator(
                    S_data, len(S_data), shuffle_batches_every_epoch=False
                    )
                S_labels = [S_lab[i] for i in S_iterator.indices]


                for feats, lengths in query_iterator:
                    lat = sesh.run(
                        [latent], feed_dict={X: feats, X_lengths: lengths}
                        )[0]

                for feats, lengths in S_iterator:
                    S_lat = sesh.run(
                        [latent], feed_dict={X: feats, X_lengths: lengths}
                        )[0]

                if normalize: 
                    latents = (lat - lat.mean(axis=0))/lat.std(axis=0)
                    s_latents = (S_lat - S_lat.mean(axis=0))/S_lat.std(axis=0)
                    if print_normalization: 
                        evaluation_library.normalization_visualization(
                            lat, latents, labels, 300, 
                            path.join(lib["output_fn"], lib["model_name"])
                            )
                else: 
                    latents = lat
                    s_latents = S_lat

                distances = cdist(latents, s_latents, "cosine")
                indexes = np.argmin(distances, axis=1)
                label_matches = few_shot_learning_library.label_matches_grid_generation_2D(query_labels, S_labels)

                for i in range(len(indexes)):
                    total += 1
                    if label_matches[i, indexes[i]]:
                        correct += 1

        return [-correct/total]
    
    if lib["train_model"]:

    	#______________________________________________________________________________________________
    	# Pre-training
    	#______________________________________________________________________________________________

    	if lib["pretrain"]:

    		print("\n" + "-"*PRINT_LENGTH)
    		print("Pre-training model")
    		print("-"*PRINT_LENGTH)

    		if lib["pretraining_model"] == "cae":
    			if lib["pretraining_data"] == "TIDigits" and lib["overwrite_pairs"] is False:

    				print("\tReading in training data pairs from {} for {}...".format(lib["pretrain_train_pair_file"], lib["pretraining_data"]))
    				pair_list = data_library.data_pairs_from_file(lib["pretrain_train_pair_file"], pretrain_train_keys)

    				if (lib["mix_training_datasets"] and (lib["pretraining_data"] != lib["other_pretraining_speech_dataset"])):
    					print("\tGenerating more training data pairs from different speakers for {}...".format(lib["other_pretraining_speech_dataset"]))
    					other_pair_list = data_library.data_pairs_from_different_speakers(other_pretrain_train_labels, other_pretrain_train_keys)
    				else: other_pair_list = []

    			else: 

    				print("\tGenerating training data pairs from different speakers for {}...".format(lib["pretraining_data"]))
    				pair_list = data_library.data_pairs_from_different_speakers(pretrain_train_labels, pretrain_train_keys)

    				if (lib["mix_training_datasets"] and (lib["pretraining_data"] != lib["other_pretraining_speech_dataset"])):
    					print("\tReading in more training data pairs from {} for {}...".format(lib["other_pretrain_train_pair_file"], lib["other_pretraining_speech_dataset"]))
    					other_pair_list = data_library.data_pairs_from_file(lib["other_pretrain_train_pair_file"], other_pretrain_train_keys)
    				else: other_pair_list = []

    		elif lib["pretraining_model"] == "ae":
    			print("\tGenerating training data pairs for {}...".format(lib["pretraining_data"]))
    			pair_list = [(i, i) for i in range(len(pretrain_train_x))]

    			if (lib["mix_training_datasets"] and (lib["pretraining_data"] != lib["other_pretraining_speech_dataset"])):
    				print("\tGenerating more training data pairs for {}...".format(lib["other_pretraining_speech_dataset"]))
    				other_pair_list = [(i, i) for i in range(len(other_pretrain_train_x))]
    			else: other_pair_list = []

    		if (lib["mix_training_datasets"] and (lib["pretraining_data"] != lib["other_pretraining_speech_dataset"])):
    			new_train_x = pretrain_train_x.copy()
    			new_train_x.extend(other_pretrain_train_x)

    			new_pair_list = pair_list.copy()
    			N = len(pair_list) if len(pair_list) > len(pretrain_train_x) else len(pretrain_train_x)
    			for (a, b) in other_pair_list:
    				new_pair_list.append((a+N, b+N))

    			new_train_mask = np.concatenate((pretrain_train_mask, other_pretrain_train_mask))

    		else:
    			new_train_x = pretrain_train_x
    			new_pair_list = pair_list
    			new_train_mask = pretrain_train_mask

    		if lib["divide_into_buckets"]:
    			train_batch_iterator = batching_library.bucketing_pair_speech_iterator(
    				new_train_x, new_pair_list, batch_size, lib["n_buckets"],
    				shuffle_batches_every_epoch=lib["shuffle_batches_every_epoch"],
    				mask_x=new_train_mask, return_mask=True
    				)
    		else:
    			train_batch_iterator = batching_library.pair_speech_iterator(
    				new_train_x, new_pair_list, batch_size,
    				shuffle_batches_every_epoch=lib["shuffle_batches_every_epoch"],
    				mask_x=new_train_mask, return_mask=True
    				)

    		model_fn = lib["intermediate_pretrain_model_fn"]

    		val_it = None
    		validation_tensor = one_shot_validation

    		pretrain_record, pretrain_log = training_library.training_model(
    			[loss, optimization, train_flag], training_placeholders, lib, train_batch_iterator,
    			lib["pretraining_epochs"], lib["patience"], lib["min_number_epochs"], lib["pretraining_model"], val_it, validation_tensor,
    			restore_fn=None, save_model_fn=lib["intermediate_pretrain_model_fn"],
    			save_best_model_fn=lib["best_pretrain_model_fn"], pretraining=True
    			)

    	#______________________________________________________________________________________________
    	# Training
    	#______________________________________________________________________________________________

    	print("\n" + "-"*PRINT_LENGTH)
    	print("Training model")
    	print("-"*PRINT_LENGTH)

    	if lib["model_type"] == "cae":
    		if lib["data_type"] == "TIDigits" and lib["overwrite_pairs"] is False:

    			print("\tReading in training data pairs from {} for {}...".format(lib["train_pair_file"], lib["data_type"]))
    			pair_list = data_library.data_pairs_from_file(lib["train_pair_file"], train_keys)

    			if (lib["mix_training_datasets"] and (lib["other_speech_dataset"] != lib["data_type"])):
    				print("\tGenerating more training data pairs from different speakers for {}...".format(lib["other_speech_dataset"]))
    				other_pair_list = data_library.data_pairs_from_different_speakers(other_train_labels, other_train_keys)
    			else: other_pair_list = []

    		else:

    			print("\tGenerating training data pairs from different speakers for {}...".format(lib["data_type"]))
    			pair_list = data_library.data_pairs_from_different_speakers(train_labels, train_keys)

    			if (lib["mix_training_datasets"] and (lib["other_speech_dataset"] != lib["data_type"])):
    				print("\tReading in more training data pairs from {} for {}...".format(lib["other_train_data_dir"], lib["other_speech_dataset"]))
    				other_pair_list = data_library.data_pairs_from_file(lib["other_train_data_dir"], other_train_keys)
    			else: other_pair_list = []

    	elif lib["model_type"] == "ae": 

    		print("\tGenerating training data pairs for {}...".format(lib["data_type"]))
    		pair_list = [(i, i) for i in range(len(train_x))]

    		if (lib["mix_training_datasets"] and (lib["other_speech_dataset"] != lib["data_type"])): 
    			print("\tGenerating more training data pairs for {}...".format(lib["other_speech_dataset"]))
    			other_pair_list = [(i, i) for i in range(len(other_train_x))]
    		else: other_pair_list = []

    	if (lib["mix_training_datasets"] and (lib["data_type"] != lib["other_speech_dataset"])):
    		new_train_x = train_x.copy()
    		new_train_x.extend(other_train_x)

    		new_pair_list = pair_list.copy()
    		N = len(pair_list) if len(pair_list) > len(train_x) else len(train_x)
    		for (a, b) in other_pair_list:
    			new_pair_list.append((a+N, b+N))

    		new_train_mask = np.concatenate((train_mask, other_train_mask))
    	else:
    		new_train_x = train_x
    		new_pair_list = pair_list
    		new_train_mask = train_x

    	if lib["divide_into_buckets"]:
    		train_batch_iterator = batching_library.bucketing_pair_speech_iterator(
    			new_train_x, new_pair_list, batch_size, lib["n_buckets"],
    			shuffle_batches_every_epoch=lib["shuffle_batches_every_epoch"],
    			mask_x=new_train_mask, return_mask=True
    			)
    	else:
    		train_batch_iterator = batching_library.pair_speech_iterator(
    			new_train_x, new_pair_list, batch_size,
    			shuffle_batches_every_epoch=lib["shuffle_batches_every_epoch"],
    			mask_x=new_train_mask, return_mask=True
    			)

    	model_fn = lib["intermediate_model_fn"]

    	val_it = None
    	validation_tensor = one_shot_validation

    	record, train_log = training_library.training_model(
    		[loss, optimization, train_flag], training_placeholders, lib, train_batch_iterator,
    		lib["epochs"], lib["patience"], lib["min_number_epochs"], lib["model_type"], val_it, validation_tensor,
    		restore_fn=lib["best_pretrain_model_fn"] if lib["pretrain"] else None,
    		save_model_fn=lib["intermediate_model_fn"], save_best_model_fn=lib["best_model_fn"]
    		)

    model_fn = model_setup_library.get_model_fn(lib)
    
    if lib["test_model"]:
        #______________________________________________________________________________________________
        #Final accuracy calculation
        #______________________________________________________________________________________________

        print("\n" + "-"*PRINT_LENGTH)
        print("Testing model")
        print("-"*PRINT_LENGTH)
        log = ""
        k = lib["K"]

        if lib["do_one_shot_test"]:

            acc = one_shot_validation(lib["one_shot_testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]

            # print(f'Accuracy of {1}-shot task: {acc}')
            print(f'\tAccuracy of {1}-shot task: {acc*100:.2f}%')
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("\tWriting: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    f'Accuracy of {1}-shot task: {acc}\n'
                    )
                write_results.write(
                    f'Accuracy of {1}-shot task: {acc*100:.2f}\n'
                    )
                write_results.close()

            log += "One-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])
            print("\n")

        if lib["do_few_shot_test"]:
            
            acc = one_shot_validation(lib["testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]

            # print(f'Accuracy of {k}-shot task: {acc}')
            print(f'\tAccuracy of {k}-shot task: {acc*100:.2f}%')
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("\tWriting: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    f'Accuracy of {k}-shot task: {acc}\n'
                    )
                write_results.write(
                    f'Accuracy of {k}-shot task: {acc*100:.2f}%\n'
                    )
                write_results.close()

            log += "{}-shot accuracy of {} at rnd_seed of {} ".format(lib["K"], acc, lib["rnd_seed"])

    print("\n" + "-"*PRINT_LENGTH)
    print("Saving model library and writing logs")
    print("-"*PRINT_LENGTH)

    if lib["train_model"]: 
        #______________________________________________________________________________________________
        # Save library
        #______________________________________________________________________________________________
        model_setup_library.save_lib(lib)

        #______________________________________________________________________________________________
        # Save records
        #______________________________________________________________________________________________

        if lib["pretrain"]: model_setup_library.save_record(lib, pretrain_record, "_pretraining")
        model_setup_library.save_record(lib, record)

        #______________________________________________________________________________________________
        # Writing model log file
        #______________________________________________________________________________________________
        
        results_fn = path.join(lib["output_fn"], lib["model_instance"]) + ".txt"
        print("\tWriting: {}".format(results_fn))
        with open(results_fn, "w") as write_results:
            if lib["pretrain"]: write_results.write(pretrain_log)
            write_results.write(train_log)
            write_results.write(log)
            write_results.close()


    print("\tWriting: {}".format(lib["model_log"]))
    with open(lib["model_log"], "a") as write_results:
        write_results.write("\n{}: ".format(lib["model_instance"]) + log)

    model_setup_library.directory_management()


def rnn_speech_classifier(lib):

    #______________________________________________________________________________________________
    # Model setup
    #______________________________________________________________________________________________

    np.random.seed(lib["rnd_seed"])
    tf.set_random_seed(lib["rnd_seed"])

    epochs = lib["epochs"]
    batch_size = lib["batch_size"] 
        
    tf.reset_default_graph()

    model_setup_library.lib_print(lib)

    #______________________________________________________________________________________________
    # Data processing
    #______________________________________________________________________________________________
    
    max_frames = lib["max_frames"]
    d_frame = lib["input_dim"]

    if lib["train_model"]:
        print("\n" + "-"*PRINT_LENGTH)
        print("Processing training data")
        print("-"*PRINT_LENGTH)
        train_x, train_labels, train_lengths, train_keys = (
            data_library.load_speech_data_from_npz(lib["train_data_dir"])
            )
        data_library.truncate_data_dim(train_x, train_lengths, d_frame, max_frames)
        train_mask = data_library.get_mask(
            train_x, train_lengths, max_frames if lib["mix_training_datasets"] else None
            )
        if lib["data_type"] == "buckeye":
            train_x, train_labels, train_lengths, train_keys = (
                data_library.remove_test_classes(
                    train_x, train_labels, train_lengths, train_keys, model_setup_library.DIGIT_LIST
                    )
                )
            data_library.test_classes(train_labels, model_setup_library.DIGIT_LIST, "training")

        if lib["pretrain"]:
            if lib["pretrain_train_data_dir"] == lib["train_data_dir"]:
                pretrain_train_x = train_x
                pretrain_train_labels = train_labels
                pretrain_train_lengths = train_lengths
                pretrain_train_keys = train_keys
                pretrain_train_mask = train_mask
            else:
                pretrain_train_x, pretrain_train_labels, pretrain_train_lengths, pretrain_train_keys = (
                    data_library.load_speech_data_from_npz(lib["pretrain_train_data_dir"])
                    )
                if lib["pretraining_data"] == "buckeye":
                    pretrain_train_x, pretrain_train_labels, pretrain_train_lengths, pretrain_train_keys = (
                        data_library.remove_test_classes(
                            pretrain_train_x, pretrain_train_labels, pretrain_train_lengths,
                            pretrain_train_keys, model_setup_library.DIGIT_LIST
                            )
                        )
                    data_library.test_classes(pretrain_train_labels, model_setup_library.DIGIT_LIST, "pretraining")

                data_library.truncate_data_dim(pretrain_train_x, pretrain_train_lengths, d_frame, max_frames)
                pretrain_train_mask = data_library.get_mask(
                    pretrain_train_x, pretrain_train_lengths, max_frames if lib["mix_training_datasets"] else None
                    )

        if (lib["mix_training_datasets"] and lib["data_type"] != lib["other_speech_dataset"]):
            if lib["other_train_data_dir"] == lib["pretrain_train_data_dir"] and lib["pretrain"]:
                other_train_x = pretrain_train_x
                other_train_labels = pretrain_train_labels
                other_train_lengths = pretrain_train_lengths
                other_train_keys = pretrain_train_keys
                other_train_mask = pretrain_train_mask
            else:
                other_train_x, other_train_labels, other_train_lengths, other_train_keys = (
                    data_library.load_speech_data_from_npz(lib["other_train_data_dir"])
                    )
                if lib["other_speech_dataset"] == "buckeye":
                    other_train_x, other_train_labels, other_train_lengths, other_train_keys = (
                        data_library.remove_test_classes(
                            other_train_x, other_train_labels, other_train_lengths, other_train_keys,
                            model_setup_library.DIGIT_LIST
                            )
                        )
                    data_library.test_classes(other_train_labels, model_setup_library.DIGIT_LIST, "other training")

                data_library.truncate_data_dim(other_train_x, other_train_lengths, d_frame, max_frames)
                other_train_mask = data_library.get_mask(other_train_x, other_train_lengths, max_frames)

            if (lib["pretrain"] and lib["pretraining_data"] != lib["other_pretraining_speech_dataset"]):
                if lib["other_pretrain_train_data_dir"] == lib["train_data_dir"]:
                    other_pretrain_train_x = train_x
                    other_pretrain_train_labels = train_labels
                    other_pretrain_train_lengths = train_lengths
                    other_pretrain_train_keys = train_keys
                    other_pretrain_train_mask = train_mask
                elif lib["other_pretrain_train_data_dir"] == lib["other_train_data_dir"]:
                    other_pretrain_train_x = other_train_x
                    other_pretrain_train_labels = other_train_labels
                    other_pretrain_train_lengths = other_train_lengths
                    other_pretrain_train_keys = other_train_keys
                    other_pretrain_train_mask = other_train_mask
                else:
                    other_pretrain_train_x, other_pretrain_train_labels, other_pretrain_train_lengths, other_pretrain_train_keys = (
                        data_library.load_speech_data_from_npz(lib["other_pretrain_train_data_dir"])
                        )
                    if lib["other_pretraining_speech_dataset"] == "buckeye":
                        other_pretrain_train_x, other_pretrain_train_labels, other_pretrain_train_lengths, other_pretrain_train_keys = (
                            data_library.remove_test_classes(
                                other_pretrain_train_x, other_pretrain_train_labels, other_pretrain_train_lengths,
                                other_pretrain_train_keys, model_setup_library.DIGIT_LIST
                                )
                            )
                        data_library.test_classes(other_pretrain_train_labels, model_setup_library.DIGIT_LIST, "other_pretraining")

                    data_library.truncate_data_dim(other_pretrain_train_x, other_pretrain_train_lengths, d_frame, max_frames)
                    other_pretrain_train_mask = data_library.get_mask(other_pretrain_train_x, other_pretrain_train_lengths, max_frames)

        print("\n" + "-"*PRINT_LENGTH)
        print("Processing validation data")
        print("-"*PRINT_LENGTH)
        val_x, val_labels, val_lengths, val_keys = (
            data_library.load_speech_data_from_npz(lib["val_data_dir"])
            )
        if (lib["validate_on_validation_dataset"] and lib["validation_speech_dataset"] == "buckeye") or (lib["validate_on_validation_dataset"] is False and lib["data_type"] == "buckeye"):
            val_x, val_labels, val_lengths, val_keys = (
                data_library.remove_test_classes(
                    val_x, val_labels, val_lengths, val_keys, model_setup_library.DIGIT_LIST
                    )
                )
            data_library.test_classes(val_labels, model_setup_library.DIGIT_LIST, "valdiation")

        data_library.truncate_data_dim(val_x, val_lengths, d_frame, max_frames)
        val_mask = data_library.get_mask(val_x, val_lengths)

        train_labels_set = list(set(train_labels))
        label_ids = {}
        for label_id, label in enumerate(sorted(train_labels_set)):
            label_ids[label] = label_id
            
        train_label_ids = []
        for label in train_labels:
            train_label_ids.append(label_ids[label])
            
        lib["num_classes"] = len(train_labels_set)

    print("\n" + "-"*PRINT_LENGTH)
    print("Processing testing data")
    print("-"*PRINT_LENGTH)
    test_x, test_labels, test_lengths, test_keys = (
        data_library.load_speech_data_from_npz(lib["test_data_dir"])
        )
    if (lib["test_on_one_shot_dataset"] and lib["one_shot_speech_dataset"] == "buckeye") or (lib["test_on_one_shot_dataset"] is False and lib["data_type"] == "buckeye"):
        test_x, test_labels, test_lengths, test_keys = (
            data_library.remove_test_classes(
                test_x, test_labels, test_lengths, test_keys, model_setup_library.DIGIT_LIST
                )
            )
        data_library.test_classes(test_labels, model_setup_library.DIGIT_LIST, "testing")

    data_library.truncate_data_dim(test_x, test_lengths, d_frame, max_frames)
    test_mask = data_library.get_mask(test_x, test_lengths)

    if lib["train_model"] is False:
        val_x = test_x
        val_labels = test_labels
        val_lengths = test_lengths
        val_keys = test_keys
    

    #______________________________________________________________________________________________
    #Building model
    #______________________________________________________________________________________________
    print("\n" + "-"*PRINT_LENGTH)
    print("RNN structure:")
    print("-"*PRINT_LENGTH)
    
    X = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
    target = tf.placeholder(tf.int32, [None, lib["num_classes"]])
    X_lengths = tf.placeholder(tf.int32, [None])
    train_flag = tf.placeholder_with_default(False, shape=())
    training_placeholders = [X, X_lengths, target]
    
    model = model_legos_library.rnn_classifier_architecture(
        [X, X_lengths], train_flag, model_setup_library.activation_lib(), lib
        )

    output = model["output"]
    latent = model["latent"]

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)
        )

    optimization = tf.train.AdamOptimizer(lib["learning_rate"]).minimize(loss)

    #_____________________________________________________________________________________________________________________________________
    # One-shot evaluation
    #______________________________________________________________________________________________

    def one_shot_validation(episode_file=lib["validation_episode_list"], data_x=val_x, data_keys=val_keys, data_labels=val_labels, normalize=True, print_normalization=False):

        episode_dict = generate_unimodal_speech_episodes.read_in_episodes(episode_file)
        correct = 0
        total = 0

        episode_numbers = np.arange(1, len(episode_dict)+1)
        np.random.shuffle(episode_numbers)

        saver = tf.train.Saver()
        with tf.Session() as sesh:
            saver.restore(sesh, model_fn)

            for episode in episode_numbers:

                episode_num = str(episode)
                query = episode_dict[episode_num]["query"]
                query_data, query_keys, query_lab = generate_unimodal_speech_episodes.episode_data(
                    query["keys"], data_x, data_keys, data_labels
                    )
                query_iterator = batching_library.speech_iterator_with_labels(
                    query_data, query_lab, len(query_data), lib["num_classes"], shuffle_batches_every_epoch=False, return_labels=False
                    )
                query_labels = [query_lab[i] for i in query_iterator.indices]

                support_set = episode_dict[episode_num]["support_set"]
                S_data, S_keys, S_lab = generate_unimodal_speech_episodes.episode_data(
                    support_set["keys"], data_x, data_keys, data_labels
                    )
                S_iterator = batching_library.speech_iterator_with_labels(
                    S_data, S_lab, len(S_data), lib["num_classes"], shuffle_batches_every_epoch=False, return_labels=False
                    )
                S_labels = [S_lab[i] for i in S_iterator.indices]


                for feats, lengths in query_iterator:
                    lat = sesh.run(
                        [output], feed_dict={X: feats, X_lengths: lengths}
                        )[0]

                for feats, lengths in S_iterator:
                    S_lat = sesh.run(
                        [output], feed_dict={X: feats, X_lengths: lengths}
                        )[0]

                if normalize: 
                    latents = (lat - lat.mean(axis=0))/lat.std(axis=0)
                    s_latents = (S_lat - S_lat.mean(axis=0))/S_lat.std(axis=0)
                    if print_normalization: 
                        evaluation_library.normalization_visualization(
                            lat, latents, labels, 300, 
                            path.join(lib["output_fn"], lib["model_name"])
                            )
                else: 
                    latents = lat
                    s_latents = S_lat

                distances = cdist(latents, s_latents, "cosine")
                indexes = np.argmin(distances, axis=1)
                label_matches = few_shot_learning_library.label_matches_grid_generation_2D(query_labels, S_labels)

                for i in range(len(indexes)):
                    total += 1
                    if label_matches[i, indexes[i]]:
                        correct += 1

        return [-correct/total]
    

    if lib["train_model"]:
        #______________________________________________________________________________________________
        #Training
        #______________________________________________________________________________________________
        
        train_batch_iterator = batching_library.speech_iterator_with_labels(
            train_x, train_label_ids, batch_size, lib["num_classes"],
            shuffle_batches_every_epoch=lib["shuffle_batches_every_epoch"]
            )

        model_fn = lib["intermediate_model_fn"]

        val_it = None
        validation_tensor = one_shot_validation
        
        record, train_log = training_library.training_model(
            [loss, optimization, train_flag], training_placeholders, lib, train_batch_iterator, 
            lib["epochs"], lib["patience"], lib["min_number_epochs"], lib["model_type"], val_it, validation_tensor, 
            restore_fn=None, 
            save_model_fn=lib["intermediate_model_fn"], save_best_model_fn=lib["best_model_fn"]
            )

    model_fn = model_setup_library.get_model_fn(lib)

    if lib["test_model"]:
        #______________________________________________________________________________________________
        #Final accuracy calculation
        #______________________________________________________________________________________________

        print("\n" + "-"*PRINT_LENGTH)
        print("Testing model")
        print("-"*PRINT_LENGTH)
        log = ""
        k = lib["K"]

        if lib["do_one_shot_test"]:

            acc = one_shot_validation(lib["one_shot_testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]

            print(f'\tAccuracy of {1}-shot task: {acc*100:.2f}%')
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("\tWriting: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    f'Accuracy of {1}-shot task: {acc}\n'
                    )
                write_results.write(
                    f'Accuracy of {1}-shot task: {acc*100:.2f}\n'
                    )
                write_results.close()

            log += "One-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])
            print("\n")

        if lib["do_few_shot_test"]:
            
            acc = one_shot_validation(lib["testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]

            print(f'\tAccuracy of {k}-shot task: {acc*100:.2f}%')
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("\tWriting: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    f'Accuracy of {k}-shot task: {acc}\n'
                    )
                write_results.write(
                    f'Accuracy of {k}-shot task: {acc*100:.2f}%\n'
                    )
                write_results.close()

            log += "{}-shot accuracy of {} at rnd_seed of {} ".format(lib["K"], acc, lib["rnd_seed"])

    if lib["train_model"]: 
        #______________________________________________________________________________________________
        # Save library
        #______________________________________________________________________________________________
        model_setup_library.save_lib(lib)

        #______________________________________________________________________________________________
        # Save records
        #______________________________________________________________________________________________

        if lib["pretrain"]: model_setup_library.save_record(lib, pretrain_record, "_pretraining")
        model_setup_library.save_record(lib, record)

        #______________________________________________________________________________________________
        # Writing model log file
        #______________________________________________________________________________________________
        
        results_fn = path.join(lib["output_fn"], lib["model_instance"]) + ".txt"
        print("\tWriting: {}".format(results_fn))
        with open(results_fn, "w") as write_results:
            if lib["pretrain"]: write_results.write(pretrain_log)
            write_results.write(train_log)
            write_results.write(log)
            write_results.close()


    print("\tWriting: {}".format(lib["model_log"]))
    with open(lib["model_log"], "a") as write_results:
        write_results.write("\n{}: ".format(lib["model_instance"]) + log)


    model_setup_library.directory_management()

def rnn_speech_siamese_model(lib):

    #______________________________________________________________________________________________
    # Model setup
    #______________________________________________________________________________________________

    np.random.seed(lib["rnd_seed"])
    tf.set_random_seed(lib["rnd_seed"])

    epochs = lib["epochs"]
    batch_size = lib["batch_size"] 
        
    tf.reset_default_graph()

    model_setup_library.lib_print(lib)

    #______________________________________________________________________________________________
    # Data processing
    #______________________________________________________________________________________________
    
    max_frames = lib["max_frames"]
    d_frame = lib["input_dim"]

    if lib["train_model"]:
        print("\n" + "-"*PRINT_LENGTH)
        print("Processing training data")
        print("-"*PRINT_LENGTH)
        train_x, train_labels, train_lengths, train_keys = (
            data_library.load_speech_data_from_npz(lib["train_data_dir"])
            )
        data_library.truncate_data_dim(train_x, train_lengths, d_frame, max_frames)
        train_mask = data_library.get_mask(
            train_x, train_lengths, max_frames if lib["mix_training_datasets"] else None
            )
        if lib["data_type"] == "buckeye":
            train_x, train_labels, train_lengths, train_keys = (
                data_library.remove_test_classes(
                    train_x, train_labels, train_lengths, train_keys, model_setup_library.DIGIT_LIST
                    )
                )
            data_library.test_classes(train_labels, model_setup_library.DIGIT_LIST, "training")

        if lib["pretrain"]:
            if lib["pretrain_train_data_dir"] == lib["train_data_dir"]:
                pretrain_train_x = train_x
                pretrain_train_labels = train_labels
                pretrain_train_lengths = train_lengths
                pretrain_train_keys = train_keys
                pretrain_train_mask = train_mask
            else:
                pretrain_train_x, pretrain_train_labels, pretrain_train_lengths, pretrain_train_keys = (
                    data_library.load_speech_data_from_npz(lib["pretrain_train_data_dir"])
                    )
                if lib["pretraining_data"] == "buckeye":
                    pretrain_train_x, pretrain_train_labels, pretrain_train_lengths, pretrain_train_keys = (
                        data_library.remove_test_classes(
                            pretrain_train_x, pretrain_train_labels, pretrain_train_lengths,
                            pretrain_train_keys, model_setup_library.DIGIT_LIST
                            )
                        )
                    data_library.test_classes(pretrain_train_labels, model_setup_library.DIGIT_LIST, "pretraining")

                data_library.truncate_data_dim(pretrain_train_x, pretrain_train_lengths, d_frame, max_frames)
                pretrain_train_mask = data_library.get_mask(
                    pretrain_train_x, pretrain_train_lengths, max_frames if lib["mix_training_datasets"] else None
                    )

        if (lib["mix_training_datasets"] and lib["data_type"] != lib["other_speech_dataset"]):
            if lib["other_train_data_dir"] == lib["pretrain_train_data_dir"] and lib["pretrain"]:
                other_train_x = pretrain_train_x
                other_train_labels = pretrain_train_labels
                other_train_lengths = pretrain_train_lengths
                other_train_keys = pretrain_train_keys
                other_train_mask = pretrain_train_mask
            else:
                other_train_x, other_train_labels, other_train_lengths, other_train_keys = (
                    data_library.load_speech_data_from_npz(lib["other_train_data_dir"])
                    )
                if lib["other_speech_dataset"] == "buckeye":
                    other_train_x, other_train_labels, other_train_lengths, other_train_keys = (
                        data_library.remove_test_classes(
                            other_train_x, other_train_labels, other_train_lengths, other_train_keys,
                            model_setup_library.DIGIT_LIST
                            )
                        )
                    data_library.test_classes(other_train_labels, model_setup_library.DIGIT_LIST, "other training")

                data_library.truncate_data_dim(other_train_x, other_train_lengths, d_frame, max_frames)
                other_train_mask = data_library.get_mask(other_train_x, other_train_lengths, max_frames)

            if (lib["pretrain"] and lib["pretraining_data"] != lib["other_pretraining_speech_dataset"]):
                if lib["other_pretrain_train_data_dir"] == lib["train_data_dir"]:
                    other_pretrain_train_x = train_x
                    other_pretrain_train_labels = train_labels
                    other_pretrain_train_lengths = train_lengths
                    other_pretrain_train_keys = train_keys
                    other_pretrain_train_mask = train_mask
                elif lib["other_pretrain_train_data_dir"] == lib["other_train_data_dir"]:
                    other_pretrain_train_x = other_train_x
                    other_pretrain_train_labels = other_train_labels
                    other_pretrain_train_lengths = other_train_lengths
                    other_pretrain_train_keys = other_train_keys
                    other_pretrain_train_mask = other_train_mask
                else:
                    other_pretrain_train_x, other_pretrain_train_labels, other_pretrain_train_lengths, other_pretrain_train_keys = (
                        data_library.load_speech_data_from_npz(lib["other_pretrain_train_data_dir"])
                        )
                    if lib["other_pretraining_speech_dataset"] == "buckeye":
                        other_pretrain_train_x, other_pretrain_train_labels, other_pretrain_train_lengths, other_pretrain_train_keys = (
                            data_library.remove_test_classes(
                                other_pretrain_train_x, other_pretrain_train_labels, other_pretrain_train_lengths,
                                other_pretrain_train_keys, model_setup_library.DIGIT_LIST
                                )
                            )
                        data_library.test_classes(other_pretrain_train_labels, model_setup_library.DIGIT_LIST, "other_pretraining")

                    data_library.truncate_data_dim(other_pretrain_train_x, other_pretrain_train_lengths, d_frame, max_frames)
                    other_pretrain_train_mask = data_library.get_mask(other_pretrain_train_x, other_pretrain_train_lengths, max_frames)

        print("\n" + "-"*PRINT_LENGTH)
        print("Processing validation data")
        print("-"*PRINT_LENGTH)
        val_x, val_labels, val_lengths, val_keys = (
            data_library.load_speech_data_from_npz(lib["val_data_dir"])
            )
        if (lib["validate_on_validation_dataset"] and lib["validation_speech_dataset"] == "buckeye") or (lib["validate_on_validation_dataset"] is False and lib["data_type"] == "buckeye"):
            val_x, val_labels, val_lengths, val_keys = (
                data_library.remove_test_classes(
                    val_x, val_labels, val_lengths, val_keys, model_setup_library.DIGIT_LIST
                    )
                )
            data_library.test_classes(val_labels, model_setup_library.DIGIT_LIST, "valdiation")

        data_library.truncate_data_dim(val_x, val_lengths, d_frame, max_frames)
        val_mask = data_library.get_mask(val_x, val_lengths)

        train_labels_set = list(set(train_labels))
        label_ids = {}
        for label_id, label in enumerate(sorted(train_labels_set)):
            label_ids[label] = label_id
            
        train_label_ids = []
        for label in train_labels:
            train_label_ids.append(label_ids[label])
        
        lib["num_classes"] = len(train_labels_set)

    print("\n" + "-"*PRINT_LENGTH)
    print("Processing testing data")
    print("-"*PRINT_LENGTH)
    test_x, test_labels, test_lengths, test_keys = (
        data_library.load_speech_data_from_npz(lib["test_data_dir"])
        )
    if (lib["test_on_one_shot_dataset"] and lib["one_shot_speech_dataset"] == "buckeye") or (lib["test_on_one_shot_dataset"] is False and lib["data_type"] == "buckeye"):
        test_x, test_labels, test_lengths, test_keys = (
            data_library.remove_test_classes(
                test_x, test_labels, test_lengths, test_keys, model_setup_library.DIGIT_LIST
                )
            )
        data_library.test_classes(test_labels, model_setup_library.DIGIT_LIST, "testing")

    data_library.truncate_data_dim(test_x, test_lengths, d_frame, max_frames)
    test_mask = data_library.get_mask(test_x, test_lengths)

    if lib["train_model"] is False:
        val_x = test_x
        val_labels = test_labels
        val_lengths = test_lengths
        val_keys = test_keys
    
    

    #______________________________________________________________________________________________
    #Building model
    #______________________________________________________________________________________________

    print("\n" + "-"*PRINT_LENGTH)
    print("RNN structure:")
    print("-"*PRINT_LENGTH)
    
    X = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
    target = tf.placeholder(tf.int32, [None])
    X_lengths = tf.placeholder(tf.int32, [None])
    train_flag = tf.placeholder_with_default(False, shape=())
    training_placeholders = [X, X_lengths, target]
    
    model = model_legos_library.siamese_rnn_architecture(
        [X, X_lengths], train_flag, model_setup_library.activation_lib(), lib
        )

    latent = model["output"]
    output = latent

    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
        labels=target, embeddings=output, margin=lib["margin"]
        )

    optimization = tf.train.AdamOptimizer(lib["learning_rate"]).minimize(loss)


    #_____________________________________________________________________________________________________________________________________
    # One-shot evaluation
    #______________________________________________________________________________________________

    def one_shot_validation(episode_file=lib["validation_episode_list"], data_x=val_x, data_keys=val_keys, data_labels=val_labels, normalize=True, print_normalization=False):

        episode_dict = generate_unimodal_speech_episodes.read_in_episodes(episode_file)
        correct = 0
        total = 0

        episode_numbers = np.arange(1, len(episode_dict)+1)
        np.random.shuffle(episode_numbers)

        saver = tf.train.Saver()
        with tf.Session() as sesh:
            saver.restore(sesh, model_fn)

            for episode in episode_numbers:

                episode_num = str(episode)
                query = episode_dict[episode_num]["query"]
                query_data, query_keys, query_lab = generate_unimodal_speech_episodes.episode_data(
                    query["keys"], data_x, data_keys, data_labels
                    )
                query_iterator = batching_library.speech_iterator_with_labels(
                    query_data, query_lab, len(query_data), lib["num_classes"], shuffle_batches_every_epoch=False, return_labels=False
                    )
                query_labels = [query_lab[i] for i in query_iterator.indices]

                support_set = episode_dict[episode_num]["support_set"]
                S_data, S_keys, S_lab = generate_unimodal_speech_episodes.episode_data(
                    support_set["keys"], data_x, data_keys, data_labels
                    )
                S_iterator = batching_library.speech_iterator_with_labels(
                    S_data, S_lab, len(S_data), lib["num_classes"], shuffle_batches_every_epoch=False, return_labels=False
                    )
                S_labels = [S_lab[i] for i in S_iterator.indices]


                for feats, lengths in query_iterator:
                    lat = sesh.run(
                        [output], feed_dict={X: feats, X_lengths: lengths}
                        )[0]

                for feats, lengths in S_iterator:
                    S_lat = sesh.run(
                        [output], feed_dict={X: feats, X_lengths: lengths}
                        )[0]

                if normalize: 
                    latents = (lat - lat.mean(axis=0))/lat.std(axis=0)
                    s_latents = (S_lat - S_lat.mean(axis=0))/S_lat.std(axis=0)
                    if print_normalization: 
                        evaluation_library.normalization_visualization(
                            lat, latents, labels, 300, 
                            path.join(lib["output_fn"], lib["model_name"])
                            )
                else: 
                    latents = lat
                    s_latents = S_lat

                distances = cdist(latents, s_latents, "cosine")
                indexes = np.argmin(distances, axis=1)
                label_matches = few_shot_learning_library.label_matches_grid_generation_2D(query_labels, S_labels)

                for i in range(len(indexes)):
                    total += 1
                    if label_matches[i, indexes[i]]:
                        correct += 1

        return [-correct/total]
    

    if lib["train_model"]:
        #______________________________________________________________________________________________
        #Training
        #______________________________________________________________________________________________

        print("\n" + "-"*PRINT_LENGTH)
        print("Training model")
        print("-"*PRINT_LENGTH)
        
        num_pairs_per_batch = lib["sample_k_examples"]
        train_x_filtered = []
        labels_set = list(set(train_labels))

        label_count = {}

        for label in train_labels:  
            id = label_ids[label]  
            if id not in label_count: label_count[id] = 0
            label_count[id] += 1

        train_label_ids = []
        for i_entry, label in enumerate(train_labels):
            id = label_ids[label]
            if label_count[id] >= num_pairs_per_batch:
              train_label_ids.append(int(label_ids[label]))
              train_x_filtered.append(train_x[i_entry])
              
        train_batch_iterator = batching_library.siamese_speech_iterator_with_one_dimensional_labels(
            train_x_filtered, train_label_ids, lib["sample_n_classes"], lib["sample_k_examples"], 
            lib["n_siamese_batches"], shuffle_batches_every_epoch=True, return_labels=True
            )

        model_fn = lib["intermediate_model_fn"]

        val_it = None
        validation_tensor = one_shot_validation
        
        record, train_log = training_library.training_model(
            [loss, optimization, train_flag], training_placeholders, lib, train_batch_iterator, 
            lib["epochs"], lib["patience"], lib["min_number_epochs"], lib["model_type"], val_it, validation_tensor, 
            restore_fn=None, 
            save_model_fn=lib["intermediate_model_fn"], save_best_model_fn=lib["best_model_fn"]
            )

    model_fn = model_setup_library.get_model_fn(lib)

    if lib["test_model"]:
        #______________________________________________________________________________________________
        #Final accuracy calculation
        #______________________________________________________________________________________________
        print("\n" + "-"*PRINT_LENGTH)
        print("Testing model")
        print("-"*PRINT_LENGTH)
        log = ""
        k = lib["K"]

        if lib["do_one_shot_test"]:

            acc = one_shot_validation(lib["one_shot_testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]

            print(f'\tAccuracy of {1}-shot task: {acc*100:.2f}%')
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("\tWriting: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    f'Accuracy of {1}-shot task: {acc}\n'
                    )
                write_results.write(
                    f'Accuracy of {1}-shot task: {acc*100:.2f}\n'
                    )
                write_results.close()

            log += "One-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])
            print("\n")

        if lib["do_few_shot_test"]:
            
            acc = one_shot_validation(lib["testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]

            print(f'Accuracy of {k}-shot task: {acc*100:.2f}%')
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("\tWriting: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    f'Accuracy of {k}-shot task: {acc}\n'
                    )
                write_results.write(
                    f'Accuracy of {k}-shot task: {acc*100:.2f}%\n'
                    )
                write_results.close()

            log += "{}-shot accuracy of {} at rnd_seed of {} ".format(lib["K"], acc, lib["rnd_seed"])

    if lib["train_model"]: 
        #______________________________________________________________________________________________
        # Save library
        #______________________________________________________________________________________________
        model_setup_library.save_lib(lib)

        #______________________________________________________________________________________________
        # Save records
        #______________________________________________________________________________________________

        if lib["pretrain"]: model_setup_library.save_record(lib, pretrain_record, "_pretraining")
        model_setup_library.save_record(lib, record)

        #______________________________________________________________________________________________
        # Writing model log file
        #______________________________________________________________________________________________

        results_fn = path.join(lib["output_fn"], lib["model_instance"]) + ".txt"
        print("\tWriting: {}".format(results_fn))
        with open(results_fn, "w") as write_results:
            if lib["pretrain"]: write_results.write(pretrain_log)
            write_results.write(train_log)
            write_results.write(log)
            write_results.close()


    print("\tWriting: {}".format(lib["model_log"]))
    with open(lib["model_log"], "a") as write_results:
        write_results.write("\n{}: ".format(lib["model_instance"]) + log)

    model_setup_library.directory_management()
