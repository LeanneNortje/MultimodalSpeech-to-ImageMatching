#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This model contains functions to setup or retrieve a model library from which the model can be 
# built, trained and tested. 
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
from paths import feats_path
from paths import data_path
from paths import general_lib_path
from paths import few_shot_lib_path
from paths import episodes
from paths import pair_path

model_path = path.join("..", model_path)
feats_path = path.join("..", feats_path)
data_path = path.join("..", data_path)
few_shot_lib_path = path.join("..", few_shot_lib_path)
pair_path = path.join("..", pair_path)

sys.path.append(path.join("..", general_lib_path))
import util_library

#_____________________________________________________________________________________________________________________________________
#
# Default library
#
#_____________________________________________________________________________________________________________________________________

default_model_lib = {


        "model_type": None,
        "architecture": None,
    
        "data_type": None,
        "features_type": None, 
        "train_tag": "None",
        "val_and_test_tag": "None",          
        "max_frames": 0,
        "input_dim": 0,

        "train_model": True,
        "test_model": True,
        "activation": "relu",

        "rnn_type": "gru",

        "epochs": 400,
        "learning_rate": 0.001,
        "keep_prob": 1.0,
        "batch_size": 300,
        "n_buckets": 3,
        "shuffle_batches_every_epoch": True,
        "divide_into_buckets": True,
        "use_one_shot_as_val": True,
        "use_one_shot_as_val_for_pretraining": True,
        "one_shot_not_few_shot": True,
        "do_one_shot_test": True,
        "do_few_shot_test": True,
        "do_clustering_etc": True,
        "kmeans": True,

        "pretrain": False,
        "pretraining_model": "ae",
        "pretrain_fn": None,
        "pretraining_epochs": 100,
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
        "margin": 0.2,
        "use_best_model": True,

        "get_pairs": False,
        
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
    parser.add_argument("--architecture", type=str, choices=["fc", "rnn"], default=default_model_lib["architecture"])
    parser.add_argument("--activation", type=str, choices=["relu", "sigmoid"], default=default_model_lib["activation"])
    parser.add_argument("--learning_rate", type=float, default=default_model_lib["learning_rate"])
    parser.add_argument("--epochs", type=int, default=default_model_lib["epochs"])
    parser.add_argument("--batch_size", type=int, default=default_model_lib["batch_size"])
    parser.add_argument("--n_buckets", type=int, default=default_model_lib["n_buckets"])
    parser.add_argument("--enc", type=str, default="default")
    parser.add_argument("--latent", type=int, default=-1)
    parser.add_argument("--latent_enc", type=str, default="default")
    parser.add_argument("--latent_func", type=str, choices=["cnn", "fc", "default"], default="default")
    parser.add_argument("--data_type", type=str, choices=["buckeye", "TIDigits", "omniglot", "MNIST"], default=default_model_lib["data_type"])
    parser.add_argument("--features_type", type=str, choices=["fbank", "mfcc", "None"], default=default_model_lib["features_type"])
    parser.add_argument("--train_tag", type=str, choices=["samediff", "samediff2", "gt", "utd", "None"], default=default_model_lib["train_tag"])
    parser.add_argument("--val_and_test_tag", type=str, choices=["gt", "None"], default=default_model_lib["val_and_test_tag"])
    parser.add_argument("--train_model", type=str, choices=["True", "False"], default=str(default_model_lib["train_model"]))
    parser.add_argument("--test_model", type=str, choices=["True", "False"], default=str(default_model_lib["test_model"]))
    parser.add_argument("--shuffle_batches_every_epoch", type=str, choices=["True", "False"], default=str(default_model_lib["shuffle_batches_every_epoch"]))
    parser.add_argument("--divide_into_buckets", type=str, choices=["True", "False"], default=str(default_model_lib["divide_into_buckets"]))
    parser.add_argument("--max_frames", type=int, default=default_model_lib["max_frames"])
    parser.add_argument("--pretrain", type=str, choices=["True", "False"], default=str(default_model_lib["pretrain"]))
    parser.add_argument("--pretraining_model", type=str, choices=["ae", "cae"], default=default_model_lib["pretraining_model"])
    parser.add_argument("--pretraining_epochs", type=int, default=default_model_lib["pretraining_epochs"])
    parser.add_argument("--use_one_shot_as_val", type=str, choices=["True", "False"], default=str(default_model_lib["use_one_shot_as_val"]))
    parser.add_argument("--use_one_shot_as_val_for_pretraining", type=str, choices=["True", "False"], default=str(default_model_lib["use_one_shot_as_val_for_pretraining"]))
    parser.add_argument("--one_shot_not_few_shot", type=str, choices=["True", "False"], default=str(default_model_lib["one_shot_not_few_shot"]))
    parser.add_argument("--do_one_shot_test", type=str, choices=["True", "False"], default=str(default_model_lib["do_one_shot_test"]))
    parser.add_argument("--use_best_model", type=str, choices=["True", "False"], default=str(default_model_lib["use_best_model"]))
    parser.add_argument("--use_best_pretrained_model", type=str, choices=["True", "False"], default=str(default_model_lib["use_best_pretrained_model"]))
    parser.add_argument("--M", type=int, default=default_model_lib["M"])
    parser.add_argument("--K", type=int, default=default_model_lib["K"])
    parser.add_argument("--Q", type=int, default=default_model_lib["Q"])
    parser.add_argument("--margin", type=int, default=default_model_lib["margin"])
    parser.add_argument("--one_shot_batches_to_use", type=str, choices=["train", "validation", "test"], default=str(default_model_lib["one_shot_batches_to_use"]))
    parser.add_argument("--one_shot_image_dataset", type=str, choices=["MNIST", "omniglot"], default=str(default_model_lib["one_shot_image_dataset"]))
    parser.add_argument("--one_shot_speech_dataset", type=str, choices=["TIDigits", "buckeye"], default=str(default_model_lib["one_shot_speech_dataset"]))
    parser.add_argument("--validation_image_dataset", type=str, choices=["MNIST", "omniglot"], default=str(default_model_lib["validation_image_dataset"]))
    parser.add_argument("--validation_speech_dataset", type=str, choices=["TIDigits", "buckeye"], default=str(default_model_lib["validation_speech_dataset"]))
    parser.add_argument("--test_on_one_shot_dataset", type=str, choices=["True", "False"], default=str(default_model_lib["test_on_one_shot_dataset"]))
    parser.add_argument("--validate_on_validation_dataset", type=str, choices=["True", "False"], default=str(default_model_lib["validate_on_validation_dataset"]))
    parser.add_argument("--kmeans", type=str, default=str(default_model_lib["kmeans"]))
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
    
    model_lib["learning_rate"] = parameters.learning_rate
    model_lib["epochs"] = parameters.epochs
    model_lib["batch_size"] = parameters.batch_size
    model_lib["n_buckets"] = parameters.n_buckets
    model_lib["data_type"] = parameters.data_type
    model_lib["features_type"] = parameters.features_type
    model_lib["train_tag"] = parameters.train_tag
    model_lib["val_and_test_tag"] = parameters.val_and_test_tag

    model_lib["train_model"] = parameters.train_model == "True"
    model_lib["test_model"] = parameters.test_model == "True"
    model_lib["train_not_test"] = model_lib["train_model"] == True
    
    model_lib["get_pairs"] = model_lib["model_type"] == "cae"
    model_lib["activation"] = parameters.activation

    model_lib["shuffle_batches_every_epoch"] = parameters.shuffle_batches_every_epoch == "True"
    model_lib["divide_into_buckets"] = parameters.divide_into_buckets == "True"
    model_lib["max_frames"] = parameters.max_frames

    model_lib["pretrain"] = parameters.pretrain == "True"
    model_lib["pretraining_model"] = parameters.pretraining_model
    model_lib["pretraining_epochs"] = parameters.pretraining_epochs
    model_lib["use_one_shot_as_val"] = parameters.use_one_shot_as_val == "True"
    model_lib["use_one_shot_as_val_for_pretraining"] = parameters.use_one_shot_as_val_for_pretraining == "True"
    model_lib["one_shot_not_few_shot"] = parameters.one_shot_not_few_shot == "True"
    model_lib["do_one_shot_test"] = parameters.do_one_shot_test == "True"
    model_lib["use_best_model"] = parameters.use_best_model
    model_lib["use_best_pretrained_model"] = parameters.use_best_pretrained_model 

    model_lib["M"] = parameters.M 
    model_lib["K"] = parameters.K
    model_lib["Q"] = parameters.Q
    model_lib["one_shot_batches_to_use"] = parameters.one_shot_batches_to_use
    model_lib["one_shot_image_dataset"] = parameters.one_shot_image_dataset
    model_lib["one_shot_speech_dataset"] = parameters.one_shot_speech_dataset
    model_lib["test_on_one_shot_dataset"] = parameters.test_on_one_shot_dataset == "True"
    model_lib["validation_image_dataset"] = parameters.validation_image_dataset
    model_lib["validation_image_dataset"] = parameters.validation_image_dataset
    model_lib["validate_on_validation_dataset"] = parameters.test_on_one_shot_dataset == "True"
    
    if model_lib["data_type"] == "buckeye" or model_lib["data_type"] == "TIDigits":

        model_lib["train_data_dir"] = path.join(
            feats_path, model_lib["data_type"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"] + ".npz"
            )
        model_lib["train_pair_file"] = path.join(
            pair_path, model_lib["data_type"], "Subsets", "Words", model_lib["features_type"], 
            model_lib["train_tag"] + "_train_" + model_lib["features_type"], "key_pairs.list"
            )
        
        if model_lib["train_tag"] == "utd": model_lib["val_and_test_tag"] = "samediff"
        else: model_lib["val_and_test_tag"] = model_lib["train_tag"]

        model_lib["val_data_dir"] = path.join(
            feats_path, model_lib["validation_speech_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "Subsets", "Words", model_lib["features_type"], model_lib["val_and_test_tag"] + "_val_" + model_lib["features_type"] + ".npz"
            )
        model_lib["val_pair_file"] = path.join(
            pair_path, model_lib["validation_speech_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "Subsets", "Words", model_lib["features_type"], model_lib["val_and_test_tag"] + "_val_" + model_lib["features_type"], 
            "key_pairs.list"
            )

        model_lib["test_data_dir"] = path.join(
            feats_path, model_lib["one_shot_speech_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], "Subsets", 
            "Words", model_lib["features_type"], model_lib["val_and_test_tag"] + "_test_" + model_lib["features_type"] + ".npz"
            )
        model_lib["test_pair_file"] = path.join(
            pair_path, model_lib["one_shot_speech_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], "Subsets", 
            "Words", model_lib["features_type"], model_lib["val_and_test_tag"] + "_test_" + model_lib["features_type"], "key_pairs.list"
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
    elif model_lib["data_type"] == "MNIST" or model_lib["data_type"] == "omniglot":
        model_lib["M"] = 10
        model_lib["input_dim"] = 28*28
        model_lib["training_on"] = "images"
        
        model_lib["train_data_dir"] = path.join(feats_path, model_lib["data_type"], "train.npz")
        model_lib["train_pair_file"] = path.join(pair_path, model_lib["data_type"], "train", "key_kmeans_pairs.list" if model_lib["kmeans"] == True else "key_pairs.list")

        model_lib["val_data_dir"] = path.join(
            feats_path, model_lib["validation_image_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "test.npz" if (model_lib["validate_on_validation_dataset"] and model_lib["validation_image_dataset"] == "omniglot") or model_lib["data_type"] == "omniglot" else "validation.npz"
            )
        model_lib["val_pair_file"] = path.join(
            pair_path, model_lib["validation_image_dataset"] if model_lib["validate_on_validation_dataset"] else model_lib["data_type"], 
            "test" if (model_lib["validate_on_validation_dataset"] and model_lib["validation_image_dataset"] == "omniglot") or model_lib["data_type"] == "omniglot" else "validation", 
            "key_kmeans_pairs.list" if model_lib["kmeans"] == True else "key_pairs.list"
            )

        model_lib["test_data_dir"] = path.join(
            feats_path, model_lib["one_shot_image_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], "test.npz"
            )
        model_lib["test_pair_file"] = path.join(
            pair_path, model_lib["one_shot_image_dataset"] if model_lib["test_on_one_shot_dataset"] else model_lib["data_type"], "test",
            "key_kmeans_pairs.list" if model_lib["kmeans"] == True else "key_pairs.list"
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

    model_lib["kmeans"] = parameters.kmeans == "True"

    if model_lib["model_type"] == "siamese": model_lib["margin"] = parameters.margin

    #_____________________________________________________________________________________________________________________________________


    model_lib["model_name"] = get_model_name(model_lib)
    date = str(datetime.now()).split(" ")
    model_lib["date"] = date
    model_lib["model_instance"] = model_lib["model_name"] + "_" + date[0] + "_" + date[1]

    model_lib["output_fn"] = path.join(model_path, model_lib["model_type"], model_lib["architecture"], path_to_model, model_lib["model_name"], model_lib["model_instance"])
    util_library.check_dir(model_lib["output_fn"])

    model_lib["best_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["model_type"] + "_best")
    model_lib["intermediate_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["model_type"] + "_last")
    
    if model_lib["pretrain"]:
        model_lib["best_pretrain_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["pretraining_model"] + "_best")
        model_lib["intermediate_pretrain_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["pretraining_model"] + "_last")
 
    model_files(model_lib)
    parameter_check(model_lib)
    model_lib["rnd_seed"] = parameters.rnd_seed

    return model_lib

#_____________________________________________________________________________________________________________________________________
#
# Setting up model architectures 
#
#_____________________________________________________________________________________________________________________________________


def model_structure(model_lib, key):
    
    all_layer_params = model_lib[key].split("_")

    model_lib[key] = [int(j) for j in all_layer_params]

def construct_dec(model_lib, key):
       
    if key == "latent_dec": 
        model_lib[key] = model_lib["latent_enc"].copy()[::-1]

    if key == "dec": model_lib[key] = model_lib["enc"].copy()[::-1]


def model_default_structure(model_lib, key):

    if model_lib["architecture"] == "fc" and model_lib["training_on"] == "images":
        if key == "enc": model_lib[key] = [500, 500]
        if key == "dec": model_lib[key] = [500, 500]
        if key == "latent": model_lib[key] = 20
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
    print("\nWriting: {}".format(lib_fn))
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
    print("{")
    for key in sorted(lib):
        print("\t{}: {}".format(key, lib[key]))
    print("}\n")

#_____________________________________________________________________________________________________________________________________
#
# Model record saving and restoring functions
#
#_____________________________________________________________________________________________________________________________________

def save_record(lib, record, prefix=""):
    record_fn = path.join(lib["output_fn"], lib["model_name"] + prefix + "_record.pkl")
    print("\nWriting: {}".format(record_fn))
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

def directory_management(log_path):

    log_file = open(log_path, 'r')
    base_dir = path.split(log_path)
    directories = os.walk(base_dir[0])
    model_name = path.splitext(base_dir[-1])[0].split("_")[0]
    instances = []
    for line in log_file:
        if re.search(str(model_name)+"_", line):
            instances.append(line.split(": ")[0])

    for root, dirs, files in directories:
        for d in dirs:
            if d not in instances:
                delete = path.join(base_dir[0], d)
                print("Removing {}".format(delete))
                shutil.rmtree(delete, ignore_errors=True)

#_____________________________________________________________________________________________________________________________________
#
# Model paramater check 
#
#_____________________________________________________________________________________________________________________________________

def parameter_check(lib):

    if (lib["training_on"] == "speech" and lib["architecture"] != "rnn") or (lib["training_on"] == "images" and lib["architecture"] != "fc"):
        print("This code only does not implement {} on {} data".format(lib["architecture"], lib["training_on"]))
        sys.exit(0)

    if ((lib["model_type"] == "ae" and lib["pretraining_model"] != "cae" and lib["pretrain"]) or (lib["model_type"] == "cae" and lib["pretraining_model"] != "ae" and lib["pretrain"])):
        if (lib["model_type"] != lib["pretraining_model"] and lib["pretrain"]):
            print("Cannot pretrain {} as {}".format(lib["model_type"], lib["pretraining_model"]))
            sys.exit(0)

    if lib["model_type"] == lib["pretraining_model"]  and lib["pretrain"]:
        print("Pretraining {} as itself, which is unnessary. Just increase the epochs to let the model train longer".format(lib["model_type"]))
        sys.exit(0)