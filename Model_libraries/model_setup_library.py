#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script contains functions to setup or retrieve a speech or image model library from which 
# the model can be built, trained and tested. 
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
import logging
import tensorflow as tf
import hashlib
import timeit
import sys
import subprocess
import pickle
from tqdm import tqdm
import re
import shutil

sys.path.append("..")
from paths import model_path
from paths import non_final_model_path
from paths import feats_path
from paths import data_path
from paths import general_lib_path
from paths import few_shot_lib_path
from paths import episodes
from paths import pair_path

model_path = path.join("..", model_path)
non_final_model_path = path.join("..", non_final_model_path)
feats_path = path.join("..", feats_path)
data_path = path.join("..", data_path)
few_shot_lib_path = path.join("..", few_shot_lib_path)
pair_path = path.join("..", pair_path)

sys.path.append(path.join("..", general_lib_path))
import util_library
PRINT_LENGTH =  180
COL_LENGTH =  PRINT_LENGTH - len("\t".expandtabs())
#_____________________________________________________________________________________________________________________________________
#
# Dataset variables
#
#_____________________________________________________________________________________________________________________________________

SPEECH_DATASETS = ["TIDigits", "buckeye"]
IMAGE_DATASETS = ["MNIST", "omniglot"]
DIGIT_LIST = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "oh"]
IMAGE_PAIR_TYPES = ["kmeans", "siamese", "classifier"]
SPEECH_PAIR_TYPES = ["siamese", "classifier"]

#_____________________________________________________________________________________________________________________________________
#
# Default library
#
#_____________________________________________________________________________________________________________________________________

default_model_lib = {


        "model_type": None,
        "architecture": None,
        "final_model": True,
    
        "data_type": None,
        "other_image_dataset": "omniglot",
        "other_speech_dataset": "buckeye",
        "features_type": None, 
        "train_tag": "None",       
        "max_frames": 0,
        "input_dim": 0,

        "mix_training_datasets": False,
        "train_model": True,
        "use_best_model": True,
        "test_model": True,

        "activation": "relu",
        "batch_size": 256,
        "n_buckets": 3,
        "margin": 0.7,
        "sample_n_classes": 88,
        "sample_k_examples": 8,
        "n_siamese_batches": 150,
        "rnn_type": "gru",
        "epochs": 50,
        "patience": 10, 
        "min_number_epochs": 10,
        "learning_rate": 0.001,
        "keep_prob": 0.9,
        "shuffle_batches_every_epoch": True,
        "divide_into_buckets": True,

        "one_shot_not_few_shot": True,
        "do_one_shot_test": True,
        "do_few_shot_test": True,
        "pair_type": "default",
        "overwrite_pairs": False,

        "pretrain": False,
        "pretraining_model": "ae",
        "pretraining_data": "None",
        "pretrain_fn": None,
        "pretraining_epochs": 50,
        "other_pretraining_image_dataset": "omniglot",
        "other_pretraining_speech_dataset": "buckeye",
        "use_best_pretrained_model": True,
        "M": 11,
        "K": 5,
        "Q": 10,
        "one_shot_batches_to_use": "test", 
        "one_shot_image_dataset": "MNIST",
        "one_shot_speech_dataset": "TIDigits", 
        "validation_image_dataset": "omniglot", 
        "validation_speech_dataset": "buckeye", 
        "test_on_one_shot_dataset": True,
        "validate_on_validation_dataset": True, 
        
        "rnd_seed": 1

    }

#_____________________________________________________________________________________________________________________________________
#
# Argument functions 
#
#_____________________________________________________________________________________________________________________________________

def arguments_for_model_restoration():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--log_path", type=str)
    return parser.parse_args()

def arguments_for_model_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["ae", "cae", "classifier", "siamese"], default=default_model_lib["model_type"])
    parser.add_argument("--architecture", type=str, choices=["cnn", "fc", "rnn"], default=default_model_lib["architecture"])
    parser.add_argument("--final_model", type=str, choices=["True", "False"], default=str(default_model_lib["final_model"]))

    parser.add_argument("--data_type", type=str, choices=["TIDigits", "buckeye", "MNIST", "omniglot"], default=default_model_lib["data_type"])
    parser.add_argument("--other_image_dataset", type=str, choices=["MNIST", "omniglot", "None"], default=default_model_lib["other_image_dataset"])
    parser.add_argument("--other_speech_dataset", type=str, choices=["TIDigits", "buckeye", "None"], default=default_model_lib["other_speech_dataset"])
    parser.add_argument("--features_type", type=str, choices=["fbank", "mfcc", "None"], default=default_model_lib["features_type"])
    parser.add_argument("--train_tag", type=str, choices=["gt", "None"], default=default_model_lib["train_tag"])
    parser.add_argument("--max_frames", type=int, default=default_model_lib["max_frames"])

    parser.add_argument("--mix_training_datasets", type=str, choices=["True", "False"], default=str(default_model_lib["mix_training_datasets"]))
    parser.add_argument("--train_model", type=str, choices=["True", "False"], default=str(default_model_lib["train_model"]))
    parser.add_argument("--use_best_model", type=str, choices=["True", "False"], default=str(default_model_lib["use_best_model"]))
    parser.add_argument("--test_model", type=str, choices=["True", "False"], default=str(default_model_lib["test_model"]))
    parser.add_argument("--activation", type=str, choices=["relu", "sigmoid"], default=default_model_lib["activation"])
    parser.add_argument("--batch_size", type=int, default=default_model_lib["batch_size"])
    parser.add_argument("--n_buckets", type=int, default=default_model_lib["n_buckets"])
    parser.add_argument("--margin", type=float, default=default_model_lib["margin"])
    parser.add_argument("--sample_n_classes", type=int, default=default_model_lib["sample_n_classes"])
    parser.add_argument("--sample_k_examples", type=int, default=default_model_lib["sample_k_examples"])
    parser.add_argument("--n_siamese_batches", type=int, default=default_model_lib["n_siamese_batches"])
    parser.add_argument("--rnn_type", type=str, choices=["gru", "lstm"], default=str(default_model_lib["rnn_type"]))
    parser.add_argument("--epochs", type=int, default=default_model_lib["epochs"])
    parser.add_argument("--learning_rate", type=float, default=default_model_lib["learning_rate"])
    parser.add_argument("--keep_prob", type=float, default=default_model_lib["keep_prob"])

    parser.add_argument("--shuffle_batches_every_epoch", type=str, choices=["True", "False"], default=str(default_model_lib["shuffle_batches_every_epoch"]))
    parser.add_argument("--divide_into_buckets", type=str, choices=["True", "False"], default=str(default_model_lib["divide_into_buckets"]))
    parser.add_argument("--one_shot_not_few_shot", type=str, choices=["True", "False"], default=str(default_model_lib["one_shot_not_few_shot"]))
    parser.add_argument("--do_one_shot_test", type=str, choices=["True", "False"], default=str(default_model_lib["do_one_shot_test"]))
    parser.add_argument("--do_few_shot_test", type=str, choices=["True", "False"], default=str(default_model_lib["do_few_shot_test"]))
    parser.add_argument("--pair_type", type=str, choices=["kmeans", "siamese", "classifier", "default"], default=str(default_model_lib["pair_type"]))
    parser.add_argument("--overwrite_pairs", type=str, choices=["True", "False"], default=str(default_model_lib["overwrite_pairs"]))

    parser.add_argument("--pretrain", type=str, choices=["True", "False"], default=str(default_model_lib["pretrain"]))
    parser.add_argument("--pretraining_model", type=str, choices=["ae", "cae"], default=default_model_lib["pretraining_model"])
    parser.add_argument("--pretraining_data", type=str, choices=["TIDigits", "buckeye", "MNIST", "omniglot", "None"], default=default_model_lib["pretraining_data"])
    parser.add_argument("--pretraining_epochs", type=int, default=default_model_lib["pretraining_epochs"])
    parser.add_argument("--other_pretraining_image_dataset", type=str, choices=["MNIST", "omniglot", "None"], default=default_model_lib["other_pretraining_image_dataset"])
    parser.add_argument("--other_pretraining_speech_dataset", type=str, choices=["TIDigits", "buckeye", "None"], default=default_model_lib["other_pretraining_speech_dataset"])
    parser.add_argument("--use_best_pretrained_model", type=str, choices=["True", "False"], default=str(default_model_lib["use_best_pretrained_model"]))

    parser.add_argument("--M", type=int, default=default_model_lib["M"])
    parser.add_argument("--K", type=int, default=default_model_lib["K"])
    parser.add_argument("--Q", type=int, default=default_model_lib["Q"])
    parser.add_argument("--one_shot_batches_to_use", type=str, choices=["train", "validation", "test"], default=str(default_model_lib["one_shot_batches_to_use"]))
    parser.add_argument("--one_shot_image_dataset", type=str, choices=["MNIST", "omniglot"], default=str(default_model_lib["one_shot_image_dataset"]))
    parser.add_argument("--one_shot_speech_dataset", type=str, choices=["TIDigits", "buckeye"], default=str(default_model_lib["one_shot_speech_dataset"]))
    parser.add_argument("--validation_image_dataset", type=str, choices=["MNIST", "omniglot"], default=str(default_model_lib["validation_image_dataset"]))
    parser.add_argument("--validation_speech_dataset", type=str, choices=["TIDigits", "buckeye"], default=str(default_model_lib["validation_speech_dataset"]))
    parser.add_argument("--test_on_one_shot_dataset", type=str, choices=["True", "False"], default=str(default_model_lib["test_on_one_shot_dataset"]))
    parser.add_argument("--validate_on_validation_dataset", type=str, choices=["True", "False"], default=str(default_model_lib["validate_on_validation_dataset"]))

    parser.add_argument("--enc", type=str, default="default")
    parser.add_argument("--latent", type=int, default=-1)
    parser.add_argument("--latent_enc", type=str, default="default")
    parser.add_argument("--latent_func", type=str, choices=["cnn", "fc", "default"], default="default")
    
    parser.add_argument("--rnd_seed", type=int, default=default_model_lib["rnd_seed"])

    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Library setup
#
#_____________________________________________________________________________________________________________________________________

def model_library_setup():

    parameters = arguments_for_model_setup()

    model_lib = default_model_lib.copy()

    model_lib["model_type"] = parameters.model_type
    model_lib["architecture"] = parameters.architecture
    model_lib["final_model"] = parameters.final_model == "True"

    model_lib["data_type"] = parameters.data_type
    model_lib["other_image_dataset"] = parameters.other_image_dataset
    if model_lib["other_image_dataset"] == "None" and model_lib["data_type"] in IMAGE_DATASETS: 
        model_lib["other_image_dataset"] = model_lib["data_type"]
    model_lib["other_speech_dataset"] = parameters.other_speech_dataset
    if model_lib["other_speech_dataset"] == "None" and model_lib["data_type"] in SPEECH_DATASETS: 
        model_lib["other_speech_dataset"] = model_lib["data_type"]
    model_lib["features_type"] = parameters.features_type
    model_lib["train_tag"] = parameters.train_tag
    model_lib["max_frames"] = parameters.max_frames

    model_lib["shuffle_batches_every_epoch"] = parameters.shuffle_batches_every_epoch == "True"
    model_lib["divide_into_buckets"] = parameters.divide_into_buckets == "True"
    model_lib["one_shot_not_few_shot"] = parameters.one_shot_not_few_shot == "True"
    model_lib["do_one_shot_test"] = parameters.do_one_shot_test == "True"
    model_lib["do_few_shot_test"] = parameters.do_few_shot_test == "True"
    model_lib["pair_type"] = parameters.pair_type
    model_lib["overwrite_pairs"] = parameters.overwrite_pairs == "True"

    model_lib["mix_training_datasets"] = parameters.mix_training_datasets == "True"
    model_lib["train_model"] = parameters.train_model == "True"
    model_lib["use_best_model"] = parameters.use_best_model
    model_lib["test_model"] = parameters.test_model == "True"
    model_lib["activation"] = parameters.activation
    model_lib["batch_size"] = parameters.batch_size
    if model_lib["batch_size"] <= 0: 
        print("Batch size must be greater than zero")
        sys.exit(0)
    model_lib["n_buckets"] = parameters.n_buckets
    if model_lib["n_buckets"] <= 0 and model_lib["divide_into_buckets"]: 
        print("Number of buckets must be greater than zero")
        sys.exit(0)
    model_lib["margin"] = parameters.margin
    if model_lib["margin"] < 0.0 and model_lib["margin"] > 1.0 and model_lib["model_type"] == "siamese": 
        print("Triplet loss margin must be between 0.0 and 1.0")
        sys.exit(0)
    model_lib["sample_n_classes"] = parameters.sample_n_classes
    model_lib["sample_k_examples"] = parameters.sample_k_examples
    model_lib["n_siamese_batches"] = parameters.n_siamese_batches
    model_lib["rnn_type"] = parameters.rnn_type
    model_lib["epochs"] = parameters.epochs
    if model_lib["epochs"] <= 0: 
        print("Epochs must be greater than zero")
        sys.exit(0)
    model_lib["learning_rate"] = parameters.learning_rate
    if model_lib["learning_rate"] <= 0.0: 
        print("Learning rate must be greater than zero")
        sys.exit(0)
    model_lib["keep_prob"] = parameters.keep_prob
    if model_lib["keep_prob"] < 0.0 and model_lib["keep_prob"] > 1.0: 
        print("Keep probability must be between 0.0 and 1.0")
        sys.exit(0)
    
    model_lib["pretrain"] = parameters.pretrain == "True"
    model_lib["pretraining_model"] = parameters.pretraining_model
    model_lib["pretraining_epochs"] = parameters.pretraining_epochs
    model_lib["pretraining_data"] = parameters.pretraining_data
    if model_lib["pretraining_data"] == "None": model_lib["pretraining_data"] = model_lib["data_type"]
    model_lib["use_best_pretrained_model"] = model_lib["use_best_model"]

    model_lib["M"] = parameters.M 
    model_lib["K"] = parameters.K
    model_lib["Q"] = parameters.Q
    model_lib["one_shot_batches_to_use"] = parameters.one_shot_batches_to_use
    model_lib["one_shot_image_dataset"] = parameters.one_shot_image_dataset
    model_lib["one_shot_speech_dataset"] = parameters.one_shot_speech_dataset
    model_lib["validation_image_dataset"] = parameters.validation_image_dataset
    model_lib["validation_speech_dataset"] = parameters.validation_speech_dataset
    model_lib["test_on_one_shot_dataset"] = parameters.test_on_one_shot_dataset == "True"
    model_lib["validate_on_validation_dataset"] = parameters.test_on_one_shot_dataset == "True"

    if model_lib["data_type"] in SPEECH_DATASETS:

        if model_lib["features_type"] == "None":
            print("The type of speech features to be used, should be specified")
            sys.exit(0)
        if model_lib["train_tag"] == "None":
            print("The speech features for isolaed words extracted in a certain manner, should be specified")
            sys.exit(0)

        model_lib["train_data_dir"] = path.join(
            feats_path, model_lib["data_type"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"] + ".npz"
            )
        model_lib["train_pair_file"] = path.join(
            pair_path, model_lib["data_type"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"], "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_pairs.list"
            )
        model_lib["pretrain_train_data_dir"] = path.join(
            feats_path, model_lib["pretraining_data"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"] + ".npz"
            )
        model_lib["pretrain_train_pair_file"] = path.join(
            pair_path, model_lib["pretraining_data"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"], "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_pairs.list"
            )

        model_lib["other_train_data_dir"] =  path.join(
            feats_path, model_lib["other_speech_dataset"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"] + ".npz"
        )
        model_lib["other_train_pair_file"] =  path.join(
            pair_path, model_lib["other_speech_dataset"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"], "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_pairs.list"
        )

        model_lib["other_pretrain_train_data_dir"] = path.join(
            feats_path, model_lib["other_pretraining_speech_dataset"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"] + ".npz"
            )
        model_lib["other_pretrain_train_pair_file"] = path.join(
            pair_path, model_lib["other_pretraining_speech_dataset"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"], "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_pairs.list"
            )
        
        model_lib["val_data_dir"] = path.join(
            feats_path, model_lib["validation_speech_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "Subsets", "Words", model_lib["features_type"], model_lib["train_tag"] + "_val_" + model_lib["features_type"] + ".npz"
            )
        model_lib["val_pair_file"] = path.join(
            pair_path, model_lib["validation_speech_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "Subsets", "Words", model_lib["features_type"], model_lib["train_tag"] + "_val_" + model_lib["features_type"], 
            "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_pairs.list"
            )

        model_lib["test_data_dir"] = path.join(
            feats_path, model_lib["one_shot_speech_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], "Subsets", 
            "Words", model_lib["features_type"], model_lib["train_tag"] + "_test_" + model_lib["features_type"] + ".npz"
            )
        model_lib["test_pair_file"] = path.join(
            pair_path, model_lib["one_shot_speech_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], "Subsets", 
            "Words", model_lib["features_type"], model_lib["train_tag"] + "_test_" + model_lib["features_type"], "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_pairs.list"
            )
        model_lib["input_dim"] = 13
        model_lib["max_frames"] = 101
        model_lib["training_on"] = "speech"
        path_to_model = path.join(model_lib["data_type"], model_lib["features_type"], model_lib["train_tag"])

        if model_lib["one_shot_not_few_shot"]: 
            model_lib["validation_episode_list"] = path.join(few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
                model_lib["M"], 1, model_lib["Q"], model_lib["validation_speech_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], "val"
                )
            )
        else: 
            model_lib["validation_episode_list"] = path.join(few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
                model_lib["M"], model_lib["K"], model_lib["Q"], model_lib["validation_speech_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], "val"
                )
            )
        model_lib["one_shot_testing_episode_list"] = path.join(few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
            model_lib["M"], 1, model_lib["Q"], model_lib["one_shot_speech_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], 
            model_lib["one_shot_batches_to_use"]
            )
        )  
        model_lib["testing_episode_list"] = path.join(few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
            model_lib["M"], model_lib["K"], model_lib["Q"], model_lib["one_shot_speech_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], 
            model_lib["one_shot_batches_to_use"]
            )
        )
    elif model_lib["data_type"] in IMAGE_DATASETS:
        model_lib["M"] -= 1
        model_lib["input_dim"] = 28*28
        model_lib["training_on"] = "images"
        
        model_lib["train_data_dir"] = path.join(feats_path, model_lib["data_type"], "train.npz")
        model_lib["train_pair_file"] = path.join(pair_path, model_lib["data_type"], "train", "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_pairs.list")

        model_lib["pretrain_train_data_dir"] = path.join(feats_path, model_lib["pretraining_data"], "train.npz")
        model_lib["pretrain_train_pair_file"] = path.join(pair_path, model_lib["pretraining_data"], "train", "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_pairs.list")

        model_lib["other_train_data_dir"] =  path.join(feats_path, model_lib["other_image_dataset"], "train.npz")
        model_lib["other_train_pair_file"] = path.join(pair_path, model_lib["other_image_dataset"], "train", "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_pairs.list")

        model_lib["other_pretrain_train_data_dir"] =  path.join(feats_path, model_lib["other_pretraining_image_dataset"], "train.npz")
        model_lib["other_pretrain_train_pair_file"] = path.join(pair_path, model_lib["other_pretraining_image_dataset"], "train", "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_pairs.list")
            
        model_lib["val_data_dir"] = path.join(
            feats_path, model_lib["validation_image_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "validation.npz"
            )
        model_lib["val_pair_file"] = path.join(
            pair_path, model_lib["validation_image_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "validation", 
            "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_pairs.list"
            )
        model_lib["val_data_dir"] = path.join(
            feats_path, model_lib["validation_image_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "validation.npz"
            )
        model_lib["val_pair_file"] = path.join(
            pair_path, model_lib["validation_image_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "validation", 
            "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_pairs.list"
            )

        model_lib["test_data_dir"] = path.join(
            feats_path, model_lib["one_shot_image_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], "test.npz"
            )
        model_lib["test_pair_file"] = path.join(
            pair_path, model_lib["one_shot_image_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], "test",
            "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_pairs.list"
            )
        
        path_to_model = path.join(model_lib["data_type"])

        if model_lib["one_shot_not_few_shot"]:
            model_lib["validation_episode_list"] = path.join(few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
                model_lib["M"], 1, model_lib["Q"], model_lib["validation_image_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], "val"
                )
            )
        else:
            model_lib["validation_episode_list"] = path.join(few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
                model_lib["M"], model_lib["K"], model_lib["Q"], model_lib["validation_image_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], "val"
                )
            )
        model_lib["one_shot_testing_episode_list"] = path.join(few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
            model_lib["M"], 1, model_lib["Q"], model_lib["one_shot_image_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], 
            model_lib["one_shot_batches_to_use"]
            )
        )  
        model_lib["testing_episode_list"] = path.join(few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
            model_lib["M"], model_lib["K"], model_lib["Q"], model_lib["one_shot_image_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], 
            model_lib["one_shot_batches_to_use"]
            )
        )

    model_lib["enc"] = parameters.enc
    if model_lib["enc"] == "default": model_default_structure(model_lib, "enc")
    else: model_structure(model_lib, "enc")

    if parameters.enc == "default": model_default_structure(model_lib, "dec")
    else: construct_dec(model_lib, "dec")
       
    model_lib["latent"] = parameters.latent
    if model_lib["latent"] == -1: model_default_structure(model_lib, "latent")
    
    model_lib["latent_enc"] = parameters.latent_enc
    if model_lib["latent_enc"] == "default": model_default_structure(model_lib, "latent_enc")
    else: model_structure(model_lib, "latent_enc")
    
    if model_lib["latent_enc"] == "default": model_default_structure(model_lib, "latent_dec")
    else: construct_dec(model_lib, "latent_dec")
    
    model_lib["latent_func"] = parameters.latent_func
    if model_lib["latent_func"] == "default": model_default_structure(model_lib, "latent_func")

#########################################################################################
    model_lib["model_name"] = get_model_name(model_lib)
    date = str(datetime.now()).split(" ")
    model_lib["date"] = date
    model_lib["model_instance"] = model_lib["model_name"] + "_" + date[0] + "_" + date[1]

    model_lib["output_fn"] = path.join(model_path if model_lib["final_model"] else non_final_model_path, 
        model_lib["pretraining_model"] + "_" + model_lib["model_type"] if model_lib["pretrain"] else model_lib["model_type"], 
        model_lib["architecture"], path_to_model, model_lib["model_name"], model_lib["model_instance"]
        )
    util_library.check_dir(model_lib["output_fn"])

    model_lib["best_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["model_type"] + "_best")
    model_lib["intermediate_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["model_type"] + "_last")
    
    if model_lib["pretrain"]:
        model_lib["best_pretrain_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["pretraining_model"] + "_best")
        model_lib["intermediate_pretrain_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["pretraining_model"] + "_last")
 
    log_dict = model_files(model_lib)
    model_lib["rnd_seed"] = parameters.rnd_seed
    if str(model_lib["rnd_seed"]) in log_dict:
        print(f'This model with seed {model_lib["rnd_seed"]} already trained.')
        directory_management()
        sys.exit(0)

    return model_lib

#_____________________________________________________________________________________________________________________________________
#
# Setting up model architectures 
#
#_____________________________________________________________________________________________________________________________________


def model_structure(model_lib, key):
    
    all_layer_params = model_lib[key].split("_")

    if model_lib["architecture"] == "cnn" and key == "enc":
        model_lib[key] = []
        model_lib["enc_strides"] = []
        model_lib["pool_layers"] = []
        model_lib["pool_func"] = "max"
        for param in all_layer_params:
            this_param = param.split(".")
            model_lib[key].append([int(this_param[0]), int(this_param[1]), int(this_param[2])])
            model_lib["enc_strides"].append([int(this_param[3]), int(this_param[4])])
            if len(this_param) == 7: model_lib["pool_layers"].append([int(this_param[5]), int(this_param[6])])
            else: model_lib["pool_layers"].append(None)
    else: model_lib[key] = [int(j) for j in all_layer_params]

def construct_dec(model_lib, key):
       
    if key == "latent_dec": 
        model_lib[key] = model_lib["latent_enc"].copy()[::-1]

    if key == "dec": 
        model_lib[key] = []
        for i in range(len(model_lib["enc"])-1, -1, -1):
            if isinstance(model_lib["enc"][i], list) is False: model_lib[key].append(model_lib["enc"][i])
            else: model_lib[key].append([model_lib["enc"][i][j] for j in range(len(model_lib["enc"][i]))])

        if model_lib["architecture"] == "cnn":
            model_lib["dec_strides"] = model_lib["enc_strides"].copy()[::-1]
            for i in range(len(model_lib[key])):
                if i != len(model_lib[key]) - 1: model_lib[key][i][-1] = model_lib[key][i+1][-1]
                else: model_lib[key][i][-1] = 1

def model_default_structure(model_lib, key):

    if model_lib["training_on"] == "images":
        if model_lib["architecture"] == "fc": 
            if key == "enc": model_lib[key] = [500, 500]
            if key == "dec": model_lib[key] = [500, 500]
            if key == "latent": model_lib[key] = 130
        elif model_lib["architecture"] == "cnn":
            if key == "enc": 
                model_lib[key] = [[3, 3, 32], [3, 3, 64], [3, 3, 128]]
                model_lib["enc_strides"] = [[1, 1], [1, 1], [1, 1]]
                model_lib["pool_layers"] = [[2, 2], [2, 2], None]
                model_lib["pool_func"] = "max"
            if key == "dec": 
                model_lib[key] = [[3, 3, 64], [3, 3, 32], [3, 3, 1]]
                model_lib["dec_strides"] = [[1, 1], [1, 1], [1, 1]]
            if key == "latent_enc": model_lib[key] = []
            if key == "latent_dec": model_lib[key] = []
            if key == "latent_func": model_lib[key] = "default"
            if key == "latent": model_lib[key] = 130
    if model_lib["architecture"] == "rnn" and model_lib["training_on"] == "speech":
        if key == "enc": model_lib[key] = [400, 400, 400]
        if key == "dec": model_lib[key] = [400, 400, 400]
        if key == "latent_func": model_lib[key] = "default"
        if key == "latent": model_lib[key] = 130
        if key == "latent_enc": model_lib[key] = []
        if key == "latent_dec": model_lib[key] = []

#_____________________________________________________________________________________________________________________________________
#
# Generating a model name
#
#_____________________________________________________________________________________________________________________________________

def get_model_name(lib):

    hasher = hashlib.md5(repr(sorted(lib.items())).encode("ascii"))
    
    return hasher.hexdigest()[:10]

#_____________________________________________________________________________________________________________________________________
#
# Model library saving, restoring and printing functions
#
#_____________________________________________________________________________________________________________________________________


def save_lib(lib):

    lib_fn = path.join(lib["output_fn"], lib["model_name"] + "_lib.pkl")
    print("\tWriting: {}".format(lib_fn))
    with open(lib_fn, "wb") as lib_write:
        pickle.dump(lib, lib_write, -1)

def restore_lib_from_arguments():
    parameters = arguments_for_model_restoration()
    lib_file = open(parameters.model_path, 'rb')
    lib = pickle.load(lib_file)
    lib["model_log"] = parameters.log_path
    util_library.check_dir(os.path.dirname(lib["model_log"]))

    if os.path.isfile(lib["model_log"]) is False:
        with open(lib["model_log"], "w") as f:
            f.write(
                "Model name: {}\nLog file was created on {} at {}\n"
                .format(lib["model_name"], lib["date"][0], lib["date"][1])
                )
            f.close()
    return lib

def restore_lib(model_path):
    lib_file = open(model_path, 'rb')
    lib = pickle.load(lib_file)
    return lib

def lib_print(lib):
    max_len = -np.inf
    for key in lib:
        if len(key) > max_len: max_len = len(key)
    max_len += 3
    print("\n" + "-"*PRINT_LENGTH)
    print("\tModel library:")
    print("-"*PRINT_LENGTH)
    tab_len = len("\t".expandtabs())
    for key in sorted(lib):
        print_key = "\t" + key + ":"
        key_value = lib[key] if lib[key] is not None else "None"

        if isinstance(key_value, (str, int, float)) is False:
            key_value = "["
            for i, val in enumerate(lib[key]):
                key_value += str(val)
                if i != len(lib[key])-1: key_value += ", "
            key_value += "]"
        elif key_value is True or key_value is False:
            if key_value: key_value = "True"
            else: key_value = "False"

        print(f'{print_key:<{max_len}}{key_value:<{PRINT_LENGTH-max_len-tab_len}}')

def command_printing(cmd):
    counter = 0
    temp_string = ""
    print("\n" + "-"*PRINT_LENGTH)
    print("\tCommand:")
    for letter in cmd:
        if counter < PRINT_LENGTH: 
            temp_string += letter
            counter += 1
        elif counter == PRINT_LENGTH:
            print(f'{temp_string}')
            temp_string = "" + letter
            counter = 1
    print(f'{temp_string}') 

#_____________________________________________________________________________________________________________________________________
#
# Model record saving and restoring functions
#
#_____________________________________________________________________________________________________________________________________

def save_record(lib, record, prefix=""):
    record_fn = path.join(lib["output_fn"], lib["model_name"] + prefix + "_record.pkl")
    print("\tWriting: {}".format(record_fn))
    with open(record_fn, "wb") as record_write:
        pickle.dump(record, record_write, -1)

def restore_record(lib, prefix=""):
    record_fn = path.join(lib["output_fn"], lib["model_name"] + prefix + "_record.pkl")
    record_file = open(record_fn, 'rb')
    record = pickle.load(record_file)
    return record

#_____________________________________________________________________________________________________________________________________
#
# Activation library
#
#_____________________________________________________________________________________________________________________________________

def activation_lib():

    activation = {}
    activation["relu"] = tf.nn.relu
    activation["sigmoid"] = tf.nn.sigmoid
    return activation

def reverse_activation_lib():

    activation = {}
    activation[tf.nn.relu] = "relu"
    activation[tf.nn.sigmoid] = "sigmoid"
    return activation

#_____________________________________________________________________________________________________________________________________
#
# Pooling library
#
#_____________________________________________________________________________________________________________________________________

def pooling_lib():

    activation = {}
    activation["max"] = tf.nn.max_pool
    activation["avg"] = tf.nn.avg_pool
    return activation

#_____________________________________________________________________________________________________________________________________
#
# Setting model_fn to restore model from 
#
#_____________________________________________________________________________________________________________________________________

def get_model_fn(lib):

    if lib["use_best_model"]: return lib["best_model_fn"]
    else: return lib["intermediate_model_fn"] 

#_____________________________________________________________________________________________________________________________________
#
# Model files and directories management
#
#_____________________________________________________________________________________________________________________________________

def model_files(lib):
    lib["model_log"] = path.join(
        path.split(lib["output_fn"])[0], lib["model_name"] + "_log.txt"
        )
    lib["model_lib_file"] = path.join(
        path.split(lib["output_fn"])[0], lib["model_name"] + "_library.txt"
        )

    if os.path.isfile(lib["model_log"]) is False:
        with open(lib["model_log"], "w") as f:
            f.write(
                "Model name: {}\nLog file was created on {} at {}\n"
                .format(lib["model_name"], lib["date"][0], lib["date"][1])
                )
            f.close()

    if os.path.isfile(lib["model_lib_file"]) is False:
        with open(lib["model_lib_file"], "w") as f:
            f.write(
                "Model library:\n"
                )
            f.write("{")
            for key in sorted(lib):
                f.write("\t{}: {}\n".format(key, lib[key]))
            f.write("}\n")
            f.close()

    if os.path.isfile(lib["model_log"]):
        log_dict = {}
        for line in open(lib["model_log"], 'r'):
            keyword = "rnd_seed of "
            if re.search(keyword, line):
                line_parts = line.strip().split(" ")
                keyword_parts = keyword.strip().split(" ")
                ind = np.where(np.asarray(line_parts) == keyword_parts[0])[0]
                rnd_seed_1 = line_parts[ind[0]+2]
                acc_1 = line_parts[ind[0]-2]
                rnd_seed_few = line_parts[ind[1]+2]
                acc_few = line_parts[ind[1]-2]
                
                if rnd_seed_1 == rnd_seed_few and rnd_seed_1 not in log_dict:
                    log_dict[rnd_seed_1] = [acc_1, acc_few]

        return log_dict
    return {}


def directory_management():

    directories = os.walk("../Model_data/")
    valid_dirs = {}
    for root, dirs, files in directories:
        for filename in files:
            if filename.split("_")[-1] == "log.txt":
                if root not in valid_dirs: valid_dirs[root] = []
                log = path.join(root, filename)
                name = root.split("/")[-1]

                log_file = open(log, 'r')

                instances = []
                for line in log_file:
                    if re.search(str(name)+"_", line):
                        instances.append(line.split(": ")[0])

                valid_dirs[root] = instances

    directories = os.walk("../Model_data/")
    for root, dirs, files in directories:
        if root in valid_dirs:
            instances = valid_dirs[root]
            for d in dirs:
                if d not in instances:
                    delete = path.join(root, d)
                    print("\tRemoving {}".format(delete))
                    shutil.rmtree(delete, ignore_errors=True)