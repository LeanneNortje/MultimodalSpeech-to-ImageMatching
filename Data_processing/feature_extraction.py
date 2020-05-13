#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This sets up the feature extraction of a spesific dataset and all the information required and 
# associated with this dataset. 
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
import logging
import tensorflow as tf

import subprocess
import sys
from tqdm import tqdm

sys.path.append("..")
from paths import data_path
data_path = path.join("..", data_path)

import speech_library

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["buckeye", "TIDigits"], 
        default="buckeye"
        )
    parser.add_argument(
        "--feat_type", type=str, choices=["fbank", "mfcc"], 
        default="fbank"
        )
    
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def dataset_library(args):

    dataset_lib = {}

    if args.dataset == "buckeye":
        dataset_lib["feats_type"] = args.feat_type
        dataset_lib["dataset"] = args.dataset

        dataset_lib["out_dir"] = args.dataset
        dataset_lib["wavs"] = path.join(data_path, args.dataset, "*", "*.wav")
        dataset_lib["vads"] = path.join(data_path, dataset_lib["dataset"], "english.wrd")
        dataset_lib["training_speakers_path"] = path.join(data_path, dataset_lib["dataset"], "devpart1_speakers.list") 
        dataset_lib["validation_speakers_path"] = path.join(data_path, dataset_lib["dataset"], "devpart2_speakers.list")
        dataset_lib["testing_speakers_path"] = path.join(data_path, dataset_lib["dataset"], "zs_speakers.list")
        
        dataset_lib["labels_to_exclude"] = ["SIL", "SPN"]
        dataset_lib["include_labels"] = True
        dataset_lib["labels_given"] = True

        dataset_lib["extract_words_or_not"] = True

    elif args.dataset == "TIDigits":
        dataset_lib["feats_type"] = args.feat_type
        dataset_lib["dataset"] = args.dataset

        dataset_lib["out_dir"] = args.dataset
        dataset_lib["wavs"] = path.join(data_path, args.dataset, "tidigits_wavs", "*", "*", "*","*.wav")
        dataset_lib["vads"] = path.join(data_path, dataset_lib["dataset"], "tidigits_fa", "words.wrd")
        dataset_lib["training_speakers_path"] = path.join(data_path, dataset_lib["dataset"], "tidigits_fa", "train_speakers.list") 
        dataset_lib["validation_speakers_path"] = path.join(data_path, dataset_lib["dataset"], "tidigits_fa", "val_speakers.list")
        dataset_lib["testing_speakers_path"] = path.join(data_path, dataset_lib["dataset"], "tidigits_fa", "test_speakers.list")

        dataset_lib["labels_to_exclude"] = []
        dataset_lib["include_labels"] = True
        dataset_lib["labels_given"] = True

        dataset_lib["extract_words_or_not"] = True

    return dataset_lib



def main():

    args = arguments()
    lib = dataset_library(args)

    feats = speech_library.extract_features(lib)

    speech_library.extract_segments(feats, lib)

if __name__ == "__main__":
    main()