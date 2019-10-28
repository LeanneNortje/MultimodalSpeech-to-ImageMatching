#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: 
#_________________________________________________________________________________________________
#
# This sript contains and encoder-decoder, classifier and Siamese models using FFNN's for imaes.
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
import generate_unimodal_image_episodes

#_____________________________________________________________________________________________________________________________________
#
# Encoder-decoder (AE and CAE) FFNN model 
#
#_____________________________________________________________________________________________________________________________________

def fc_vision_encdec(lib):

    #______________________________________________________________________________________________
    # Model setup
    #______________________________________________________________________________________________

    np.random.seed(lib["rnd_seed"])
    tf.set_random_seed(lib["rnd_seed"])

    epochs = lib["epochs"]
    batch_size = lib["batch_size"] 
        
    tf.reset_default_graph()
    print("\n" + "-"*150)

    #______________________________________________________________________________________________
    # Data processing
    #______________________________________________________________________________________________

    print("\nModel parameters:")
    model_setup_library.lib_print(lib)

    train_x, train_labels, train_keys = (
        data_library.load_image_data_from_npz(lib["train_data_dir"])
        )
    val_x, val_labels, val_keys = (
        data_library.load_image_data_from_npz(lib["val_data_dir"])
        )
    test_x, test_labels, test_keys = (
        data_library.load_image_data_from_npz(lib["test_data_dir"])
        )
  
    print("\n" + "-"*150)

    #______________________________________________________________________________________________
    #Building model
    #______________________________________________________________________________________________

    print("FFNN structure:")
     
    X = tf.placeholder(tf.float32, [None, lib["input_dim"]])
    target = tf.placeholder(tf.float32, [None, lib["input_dim"]])

    model = model_legos_library.fc_architecture(
        [X], lib["enc"], lib["latent"], lib["dec"], lib, model_setup_library.activation_lib(), 0
        )
    
    output = model["output"]
    latent = model["latent"]

    output = tf.nn.sigmoid(output) # ensure values are between 0 and 1 to compare to target image
    loss = tf.reduce_mean(tf.pow(target - output, 2))

    optimization = tf.train.AdamOptimizer(lib["learning_rate"]).minimize(loss)

    print("\n" + "-"*150)

    #_____________________________________________________________________________________________________________________________________
    # One-shot evaluation
    #______________________________________________________________________________________________

    def one_shot_validation(episode_file=lib["validation_episode_list"], data_x=val_x, data_keys=val_keys, data_labels=val_labels, normalize=True, print_normalization=False):

        episode_dict = generate_unimodal_image_episodes.read_in_episodes(episode_file)
        correct = 0
        total = 0

        # np.random.seed(lib["rnd_seed"])
        episode_numbers = np.arange(1, len(episode_dict)+1)
        np.random.shuffle(episode_numbers)

        saver = tf.train.Saver()
        with tf.Session() as sesh:
            saver.restore(sesh, model_fn)

            for episode in episode_numbers:

                episode_num = str(episode)
                query = episode_dict[episode_num]["query"]
                query_data, query_keys, query_lab = generate_unimodal_image_episodes.episode_data(
                    query["keys"], data_x, data_keys, data_labels
                    )
                query_iterator = batching_library.image_iterator(
                    query_data, len(query_data), shuffle_batches_every_epoch=False
                    )
                query_labels = [query_lab[i] for i in query_iterator.indices]

                support_set = episode_dict[episode_num]["support_set"]
                S_data, S_keys, S_lab = generate_unimodal_image_episodes.episode_data(
                    support_set["keys"], data_x, data_keys, data_labels
                    )
                S_iterator = batching_library.image_iterator(
                    S_data, len(S_data), shuffle_batches_every_epoch=False
                    )
                S_labels = [S_lab[i] for i in S_iterator.indices]


                for feats in query_iterator:
                    lat = sesh.run(
                        [latent], feed_dict={X: feats}
                        )[0]

                for feats in S_iterator:
                    S_lat = sesh.run(
                        [latent], feed_dict={X: feats}
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
            if lib["pretraining_model"] == "cae": 
                
                if lib["data_type"] == "MNIST": pair_list = data_library.data_pairs_from_file(lib["train_pair_file"], train_keys)
                else: pair_list = data_library.all_data_pairs(train_labels)

                if lib["validate_on_validation_dataset"] and lib["validation_image_dataset"] == "omniglot": val_pair_list = data_library.all_data_pairs(val_labels)
                elif lib["data_type"] == "omniglot": val_pair_list = data_library.all_data_pairs(val_labels)
                else: val_pair_list = data_library.data_pairs_from_file(lib["val_pair_file"], val_keys)
                    
            elif lib["pretraining_model"] == "ae": 
                pair_list = [(i, i) for i in range(len(train_x))]
                val_pair_list = [(i, i) for i in range(len(val_x))]
                

            train_batch_iterator = batching_library.pair_image_iterator(
                train_x, pair_list, batch_size,  
                lib["shuffle_batches_every_epoch"]
                )

            model_fn = lib["intermediate_pretrain_model_fn"]

            if lib["use_one_shot_as_val_for_pretraining"]:
                val_it = None
                validation_tensor = one_shot_validation
            else:
                val_it = batching_library.pair_image_iterator(
                    val_x, val_pair_list, batch_size, lib["shuffle_batches_every_epoch"]
                    )
                validation_tensor = None
            
            pretrain_record, pretrain_log = training_library.training_model(
                [loss, optimization], [X, target], lib, train_batch_iterator, 
                lib["pretraining_epochs"], lib["pretraining_model"], val_it, validation_tensor, restore_fn=None, 
                save_model_fn=lib["intermediate_pretrain_model_fn"], 
                save_best_model_fn=lib["best_pretrain_model_fn"]
                )

            print("\n" + "-"*150)

        #______________________________________________________________________________________________
        #Training
        #______________________________________________________________________________________________
        # if lib["train_model"]:
        if lib["model_type"] == "cae":

            if lib["data_type"] == "MNIST": pair_list = data_library.data_pairs_from_file(lib["train_pair_file"], train_keys)
            else: pair_list = data_library.data_pairs(train_labels)

            if lib["validate_on_validation_dataset"] and lib["validation_image_dataset"] == "omniglot": val_pair_list = data_library.all_data_pairs(val_labels)
            elif lib["data_type"] == "omniglot": val_pair_list = data_library.all_data_pairs(val_labels)
            else: val_pair_list = data_library.data_pairs_from_file(lib["val_pair_file"], val_keys)

        elif lib["model_type"] == "ae": 
            pair_list = [(i, i) for i in range(len(train_x))]
            val_pair_list = [(i, i) for i in range(len(val_x))]

        train_batch_iterator = batching_library.pair_image_iterator(
            train_x, pair_list, batch_size,  
            lib["shuffle_batches_every_epoch"]
            )

        model_fn = lib["intermediate_model_fn"]

        if lib["use_one_shot_as_val"]:
            val_it = None
            validation_tensor = one_shot_validation
        else:
            val_it = batching_library.pair_image_iterator(
                val_x, val_pair_list, batch_size, lib["shuffle_batches_every_epoch"]
                )
            validation_tensor = None
        
        record, train_log = training_library.training_model(
            [loss, optimization], [X, target], lib, train_batch_iterator, 
            lib["epochs"], lib["model_type"], val_it, validation_tensor, 
            restore_fn=lib["best_pretrain_model_fn"] if lib["pretrain"] else None, 
            save_model_fn=lib["intermediate_model_fn"], save_best_model_fn=lib["best_model_fn"]
            )

        print("\n" + "-"*150)

    model_fn = model_setup_library.get_model_fn(lib)
    results = []
    if lib["test_model"]:
    #______________________________________________________________________________________________
    #Final accuracy calculation
    #______________________________________________________________________________________________
        #log = "\n{}: ".format(lib["model_instance"])
        log = ""
        if lib["do_one_shot_test"]:

            acc = one_shot_validation(lib["one_shot_testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]
            print("Accuracy of one-shot task: {}".format(acc))
            print("Accuracy of one-shot task: {}%".format(acc*100))
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("Writing: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    "Accuracy of one shot-ask: {}\n".format(acc)
                    )
                write_results.write(
                    "Accuracy of one shot-task: {}%\n".format(acc*100)
                    )
                write_results.close()
            print("\n" + "-"*150)

            log += "One-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])
            results.append(acc)

        if lib["do_few_shot_test"]:
            
            acc = one_shot_validation(lib["testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]
            print("Accuracy of {}-shot task: {}".format(lib["K"], acc))
            print("Accuracy of {}-shot task: {}%".format(lib["K"], acc*100))
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("Writing: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    "Accuracy of {} shot task: {}\n".format(lib["K"], acc)
                    )
                write_results.write(
                    "Accuracy of {}-shot task: {}%\n".format(lib["K"], acc*100)
                    )
                write_results.close()
            print("\n" + "-"*150)

            log += "{}-shot accuracy of {} at rnd_seed of {} ".format(lib["K"], acc, lib["rnd_seed"])
            results.append(acc)

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
        # Writing model log files
        #______________________________________________________________________________________________
        
        results_fn = path.join(lib["output_fn"], lib["model_instance"]) + ".txt"
        print("Writing: {}".format(results_fn))
        with open(results_fn, "w") as write_results:
            if lib["pretrain"]: write_results.write(pretrain_log)
            write_results.write(train_log)
            write_results.write(log)
            write_results.close()
        print("\n" + "-"*150)


    print("Writing: {}".format(lib["model_log"]))
    with open(lib["model_log"], "a") as write_results:
        write_results.write("\n{}: ".format(lib["model_instance"]) + log)
    print("\n" + "-"*150)

    model_setup_library.directory_management(lib["model_log"])

#_____________________________________________________________________________________________________________________________________
#
# Classifier FFNN model
#
#_____________________________________________________________________________________________________________________________________

def fc_vision_classifier(lib):

    #______________________________________________________________________________________________
    # Model setup
    #______________________________________________________________________________________________

    np.random.seed(lib["rnd_seed"])
    tf.set_random_seed(lib["rnd_seed"])

    epochs = lib["epochs"]
    batch_size = lib["batch_size"] 
        
    tf.reset_default_graph()
    print("\n" + "-"*150)

    #______________________________________________________________________________________________
    # Data processing
    #______________________________________________________________________________________________

    print("\nModel parameters:")
    model_setup_library.lib_print(lib)

    train_x, train_labels, train_keys = (
        data_library.load_image_data_from_npz(lib["train_data_dir"])
        )
    val_x, val_labels, val_keys = (
        data_library.load_image_data_from_npz(lib["val_data_dir"])
        )
    test_x, test_labels, test_keys = (
        data_library.load_image_data_from_npz(lib["test_data_dir"])
        )

    # Convert labels to integer numbers for training
    train_labels_set = list(set(train_labels))
    label_ids = {}
    for label_id, label in enumerate(sorted(train_labels_set)):
        label_ids[label] = label_id
    train_label_ids = []
    for label in train_labels:
        train_label_ids.append(label_ids[label])

    lib["num_classes"] = len(train_labels_set)

    print("\n" + "-"*150)

    #______________________________________________________________________________________________
    #Building model
    #______________________________________________________________________________________________

    print("FFNN structure:")
     
    X = tf.placeholder(tf.float32, [None, lib["input_dim"]])
    target = tf.placeholder(tf.int32, [None, lib["num_classes"]])
    training_placeholders = [X, target]

    model = model_legos_library.fc_classifier_architecture(
        [X], lib["enc"], lib["latent"], lib, model_setup_library.activation_lib()
        )
    
    output = model["output"]
    latent = model["latent"]

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)
        )

    optimization = tf.train.AdamOptimizer(lib["learning_rate"]).minimize(loss)

    print("\n" + "-"*150)

    #_____________________________________________________________________________________________________________________________________
    # One-shot evaluation
    #______________________________________________________________________________________________

    def one_shot_validation(episode_file=lib["validation_episode_list"], data_x=val_x, data_keys=val_keys, data_labels=val_labels, normalize=True, print_normalization=False):

        episode_dict = generate_unimodal_image_episodes.read_in_episodes(episode_file)
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
                query_data, query_keys, query_lab = generate_unimodal_image_episodes.episode_data(
                    query["keys"], data_x, data_keys, data_labels
                    )
                query_iterator = batching_library.image_iterator_with_labels(
                    query_data, query_lab, len(query_data), lib["num_classes"], shuffle_batches_every_epoch=False, return_labels=False
                    )
                query_labels = [query_lab[i] for i in query_iterator.indices]

                support_set = episode_dict[episode_num]["support_set"]
                S_data, S_keys, S_lab = generate_unimodal_image_episodes.episode_data(
                    support_set["keys"], data_x, data_keys, data_labels
                    )
                S_iterator = batching_library.image_iterator_with_labels(
                    S_data, S_lab, len(S_data), lib["num_classes"], shuffle_batches_every_epoch=False, return_labels=False
                    )
                S_labels = [S_lab[i] for i in S_iterator.indices]


                for feats in query_iterator:
                    lat = sesh.run(
                        [latent], feed_dict={X: feats}
                        )[0]

                for feats in S_iterator:
                    S_lat = sesh.run(
                        [latent], feed_dict={X: feats}
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

        train_batch_iterator = batching_library.image_iterator_with_labels(
            train_x, train_label_ids, batch_size, lib["num_classes"],
            lib["shuffle_batches_every_epoch"]
            )

        model_fn = lib["intermediate_model_fn"]

        val_it = None
        validation_tensor = one_shot_validation
        
        record, train_log = training_library.training_model(
            [loss, optimization], [X, target], lib, train_batch_iterator, 
            lib["epochs"], lib["model_type"], val_it, validation_tensor, 
            restore_fn=None, 
            save_model_fn=lib["intermediate_model_fn"], save_best_model_fn=lib["best_model_fn"]
            )

        print("\n" + "-"*150)

    model_fn = model_setup_library.get_model_fn(lib)
    results = []
    if lib["test_model"]:
    #______________________________________________________________________________________________
    #Final accuracy calculation
    #______________________________________________________________________________________________

        log = ""
        if lib["do_one_shot_test"]:

            acc = one_shot_validation(lib["one_shot_testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]
            print("Accuracy of one-shot task: {}".format(acc))
            print("Accuracy of one-shot task: {}%".format(acc*100))
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("Writing: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    "Accuracy of one shot-ask: {}\n".format(acc)
                    )
                write_results.write(
                    "Accuracy of one shot-task: {}%\n".format(acc*100)
                    )
                write_results.close()
            print("\n" + "-"*150)

            log += "One-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])
            results.append(acc)

        if lib["do_few_shot_test"]:
            
            acc = one_shot_validation(lib["testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]
            print("Accuracy of {}-shot task: {}".format(lib["K"], acc))
            print("Accuracy of {}-shot task: {}%".format(lib["K"], acc*100))
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("Writing: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    "Accuracy of {} shot task: {}\n".format(lib["K"], acc)
                    )
                write_results.write(
                    "Accuracy of {}-shot task: {}%\n".format(lib["K"], acc*100)
                    )
                write_results.close()
            print("\n" + "-"*150)

            log += "{}-shot accuracy of {} at rnd_seed of {} ".format(lib["K"], acc, lib["rnd_seed"])
            results.append(acc)

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
        # Writing model log files
        #______________________________________________________________________________________________
        
        results_fn = path.join(lib["output_fn"], lib["model_instance"]) + ".txt"
        print("Writing: {}".format(results_fn))
        with open(results_fn, "w") as write_results:
            if lib["pretrain"]: write_results.write(pretrain_log)
            write_results.write(train_log)
            write_results.write(log)
            write_results.close()
        print("\n" + "-"*150)


    print("Writing: {}".format(lib["model_log"]))
    with open(lib["model_log"], "a") as write_results:
        write_results.write("\n{}: ".format(lib["model_instance"]) + log)
    print("\n" + "-"*150)

    model_setup_library.directory_management(lib["model_log"])

#_____________________________________________________________________________________________________________________________________
#
# Siamese FFNN model
#
#_____________________________________________________________________________________________________________________________________

def fc_vision_siamese(lib):

    #______________________________________________________________________________________________
    # Model setup
    #______________________________________________________________________________________________

    np.random.seed(lib["rnd_seed"])
    tf.set_random_seed(lib["rnd_seed"])

    epochs = lib["epochs"]
    batch_size = lib["batch_size"] 
        
    tf.reset_default_graph()
    print("\n" + "-"*150)

    #______________________________________________________________________________________________
    # Data processing
    #______________________________________________________________________________________________

    print("\nModel parameters:")
    model_setup_library.lib_print(lib)

    train_x, train_labels, train_keys = (
        data_library.load_image_data_from_npz(lib["train_data_dir"])
        )
    val_x, val_labels, val_keys = (
        data_library.load_image_data_from_npz(lib["val_data_dir"])
        )
    test_x, test_labels, test_keys = (
        data_library.load_image_data_from_npz(lib["test_data_dir"])
        )

    # Convert labels to integer numbers for training
    train_labels_set = list(set(train_labels))
    label_ids = {}
    for label_id, label in enumerate(sorted(train_labels_set)):
        label_ids[label] = label_id
    train_label_ids = []
    for label in train_labels:
        train_label_ids.append(label_ids[label])

    lib["num_classes"] = len(train_labels_set)
  
    print("\n" + "-"*150)

    #______________________________________________________________________________________________
    #Building model
    #______________________________________________________________________________________________

    print("FFNN structure:")
     
    X = tf.placeholder(tf.float32, [None, lib["input_dim"]])
    target = tf.placeholder(tf.int32, [None])
    training_placeholders = [X, target]

    model = model_legos_library.siamese_fc_architecture(
        [X], lib, model_setup_library.activation_lib()
        )
    
    latent = tf.nn.l2_normalize(model["output"], axis=1)
    output = latent

    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
        labels=target, embeddings=output, margin=lib["margin"]
        )

    optimization = tf.train.AdamOptimizer(lib["learning_rate"]).minimize(loss)

    print("\n" + "-"*150)

    #_____________________________________________________________________________________________________________________________________
    # One-shot evaluation
    #______________________________________________________________________________________________

    def one_shot_validation(episode_file=lib["validation_episode_list"], data_x=val_x, data_keys=val_keys, data_labels=val_labels, normalize=True, print_normalization=False):

        episode_dict = generate_unimodal_image_episodes.read_in_episodes(episode_file)
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
                query_data, query_keys, query_lab = generate_unimodal_image_episodes.episode_data(
                    query["keys"], data_x, data_keys, data_labels
                    )
                query_iterator = batching_library.image_iterator_with_labels(
                    query_data, query_lab, len(query_data), lib["num_classes"], shuffle_batches_every_epoch=False, return_labels=False
                    )
                query_labels = [query_lab[i] for i in query_iterator.indices]

                support_set = episode_dict[episode_num]["support_set"]
                S_data, S_keys, S_lab = generate_unimodal_image_episodes.episode_data(
                    support_set["keys"], data_x, data_keys, data_labels
                    )
                S_iterator = batching_library.image_iterator_with_labels(
                    S_data, S_lab, len(S_data), lib["num_classes"], shuffle_batches_every_epoch=False, return_labels=False
                    )
                S_labels = [S_lab[i] for i in S_iterator.indices]


                for feats in query_iterator:
                    lat = sesh.run(
                        [latent], feed_dict={X: feats}
                        )[0]

                for feats in S_iterator:
                    S_lat = sesh.run(
                        [latent], feed_dict={X: feats}
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

        if lib["data_type"] == "omniglot": 
            pair_list = data_library.data_pairs(train_label_ids)
        else: 
            pair_list = [(i, i) for i in range(len(train_label_ids))]
           
        train_batch_iterator = batching_library.image_iterator_with_one_dimensional_labels(
            train_x, pair_list, train_label_ids, batch_size, lib["shuffle_batches_every_epoch"]
            )

        model_fn = lib["intermediate_model_fn"]

        val_it = None
        validation_tensor = one_shot_validation
        
        record, train_log = training_library.training_model(
            [loss, optimization], [X, target], lib, train_batch_iterator, 
            lib["epochs"], lib["model_type"], val_it, validation_tensor, 
            restore_fn=None, 
            save_model_fn=lib["intermediate_model_fn"], save_best_model_fn=lib["best_model_fn"]
            )

        print("\n" + "-"*150)

    model_fn = model_setup_library.get_model_fn(lib)
    results = []
    if lib["test_model"]:
    #______________________________________________________________________________________________
    #Final accuracy calculation
    #______________________________________________________________________________________________

        log = ""
        if lib["do_one_shot_test"]:

            acc = one_shot_validation(lib["one_shot_testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]
            print("Accuracy of one-shot task: {}".format(acc))
            print("Accuracy of one-shot task: {}%".format(acc*100))
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("Writing: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    "Accuracy of one shot-ask: {}\n".format(acc)
                    )
                write_results.write(
                    "Accuracy of one shot-task: {}%\n".format(acc*100)
                    )
                write_results.close()
            print("\n" + "-"*150)

            log += "One-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])
            results.append(acc)

        if lib["do_few_shot_test"]:
            
            acc = one_shot_validation(lib["testing_episode_list"], test_x, test_keys, test_labels, normalize=True)
            acc = -acc[0]
            print("Accuracy of {}-shot task: {}".format(lib["K"], acc))
            print("Accuracy of {}-shot task: {}%".format(lib["K"], acc*100))
            results_fn = path.join(lib["output_fn"], lib["model_name"]) + "_one_shot_learning_results.txt"
            print("Writing: {}".format(results_fn))
            with open(results_fn, "w") as write_results:
                write_results.write(
                    "Accuracy of {} shot task: {}\n".format(lib["K"], acc)
                    )
                write_results.write(
                    "Accuracy of {}-shot task: {}%\n".format(lib["K"], acc*100)
                    )
                write_results.close()
            print("\n" + "-"*150)

            log += "{}-shot accuracy of {} at rnd_seed of {} ".format(lib["K"], acc, lib["rnd_seed"])
            results.append(acc)

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
        # Writing model log files
        #______________________________________________________________________________________________
        
        results_fn = path.join(lib["output_fn"], lib["model_instance"]) + ".txt"
        print("Writing: {}".format(results_fn))
        with open(results_fn, "w") as write_results:
            if lib["pretrain"]: write_results.write(pretrain_log)
            write_results.write(train_log)
            write_results.write(log)
            write_results.close()
        print("\n" + "-"*150)


    print("Writing: {}".format(lib["model_log"]))
    with open(lib["model_log"], "a") as write_results:
        write_results.write("\n{}: ".format(lib["model_instance"]) + log)
    print("\n" + "-"*150)

    model_setup_library.directory_management(lib["model_log"])
