#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script contains the building blocks to extract speech features, as well as supporting 
# functions to these blocks.
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

import subprocess
import sys
from tqdm import tqdm

from scipy.fftpack import dct

import features_library
from features_library import my_rounding
from features_library import filterbank
from features_library import mfcc

sys.path.append("..")
from paths import data_path
from paths import general_lib_path
from paths import model_lib_path
data_path = path.join("..", data_path)

sys.path.append(path.join("..", model_lib_path))
import model_setup_library

sys.path.append(path.join("..", general_lib_path))
import util_library

PRINT_LENGTH = model_setup_library.PRINT_LENGTH

#_____________________________________________________________________________________________________________________________________
#
# Utils
#
#_____________________________________________________________________________________________________________________________________

def write_feats(feats, dir):
    print("Writing: {}".format(dir))
    np.savez_compressed(dir, **feats)

#_____________________________________________________________________________________________________________________________________
#
# Feature extraction
#
#_____________________________________________________________________________________________________________________________________

def extract_features(lib, **kwargs):

    dataset = lib["dataset"]
    out_dir = lib["out_dir"]
    wavs = lib["wavs"]
    feats_type = lib["feats_type"]

    util_library.check_dir(out_dir)

    feats = {}
    print("\nExtracting features:")
    for wav_fn in tqdm(sorted(glob.glob(wavs))):
        samplerate, signal = wavfile.read(wav_fn)

        if feats_type == "fbank":
            wanted_feats = filterbank(
                signal, sampling_frequency=samplerate, preemphasis_fact=0.97, winlen=0.025, 
                winstep=0.01, winfunc=np.hamming, lowf=0, highf=None, nfft=None, nfilt=45,  
                return_energy=False, **kwargs
                )

        elif feats_type == "mfcc":
            wanted_feats = mfcc(
                signal, sampling_frequency=samplerate, preemphasis_fact=0.97, winlen=0.025, 
                winstep=0.01, winfunc=np.hamming, lowf=0, highf=None, nfft=None, nfilt=24, 
                numcep=13, ceplifter=22, append_energy=True, **kwargs
                )
        if lib["dataset"] == "TIDigits": 
            parts = wav_fn.strip().split("/")
            save_key = parts[-2] + "-" + parts[-1].strip().split(".")[0]
        else: save_key = path.splitext(path.split(wav_fn)[-1])[0]
        
        feats[save_key] = wanted_feats
    feats_dir = path.join(out_dir, "Features", "Raw", feats_type)
    util_library.check_dir(feats_dir)
    raw_feats_dir = path.join(feats_dir, dataset + "_" + feats_type + "_features")
    speakers = features_library.get_speakers(feats)
    mean, variance = features_library.speaker_mean_and_variance(feats, speakers)
    feats = features_library.speaker_mean_variance_normalization(feats, mean, variance)
    write_feats(feats, raw_feats_dir)

    print("\n" + "-"*PRINT_LENGTH)

    return feats

#_____________________________________________________________________________________________________________________________________
#
# Extracting segments
#
#_____________________________________________________________________________________________________________________________________

def extract_segments(feats, lib, **kwargs):

    dataset = lib["dataset"]
    out_dir = lib["out_dir"]
    feats_type = lib["feats_type"]
    vads = lib["vads"]
    labels_to_exclude = lib["labels_to_exclude"]
    include_labels = lib["include_labels"]
    segments_or_words = "Segments"
    extract_words_or_not = lib["extract_words_or_not"]
    labels_given = lib["labels_given"]

    util_library.check_dir(out_dir)

    segment_path = path.join(out_dir, "Features", "Segments", feats_type)
    util_library.check_dir(segment_path)
    
    segments = segment_regions(
        feats, vads, include_labels, labels_to_exclude, labels_given, **kwargs
        )

    seg_info_dir = path.join(segment_path, dataset + "_" + feats_type + "_segments_list")
    write_feats(segments, seg_info_dir)

    segmented_feats = get_segments(feats, segments)
    
    seg_feats_dir = path.join(segment_path, dataset + "_" + feats_type + "_segmented_features")
    write_feats(segmented_feats, seg_feats_dir)
    
    print("\n" + "-"*PRINT_LENGTH)
    
    train_feats, train_list, val_feats, val_list, test_feats, test_list = extract_subsets(
        lib, segments_or_words, **kwargs
        ) 
    
    print("\n" + "-"*PRINT_LENGTH)

    if extract_words_or_not:
        print("\nExtracting ground truth words from subset segments:\n")
        extract_words(train_feats, train_list, lib, "train", **kwargs)
        extract_words(val_feats, val_list, lib, "val", **kwargs)
        extract_words(test_feats, test_list, lib, "test", **kwargs)
        print("\n" + "-"*PRINT_LENGTH)

#_____________________________________________________________________________________________________________________________________
#
# Acquiring segment regions
#
#_____________________________________________________________________________________________________________________________________

def segment_regions(feats, vad_file, include_labels=False, labels_to_exclude=[], labels_given=False, **kwargs):

    print("\nExtracting segment regions:\n")

    output_dict = {}
    segment_info = []

    prev_key = ""
    prev_token_label = ""
    prev_end_time = -1
    start_time = -1

    if include_labels:

        with open(vad_file, "r") as vads:
            for line in vads:


                if labels_given: 
                    key, start, end, label = line.strip(
                        ).split()
                else: 
                    print("Labels not given but expected in segment dictionary")
                    continue
                
                start = float(start)
                end = float(end)
                key = key.replace("_", "-")

                if label in labels_to_exclude:
                    continue
                if prev_end_time != start or prev_key != key:
                    if prev_end_time != -1: 
                        start_frame = int(start_time*100)
                        prev_end_frame = int(prev_end_time*100)+1
                        
                        if prev_end_frame > feats[prev_key].shape[0]:
                            prev_end_frame = feats[prev_key].shape[0]
                        feat_key = prev_key + "_{:06d}-{:06d}".format(start_frame, prev_end_frame)
                        output_dict[feat_key] = segment_info
                    segment_info = []
                    start_time = start
                prev_end_time = end
                prev_token_label = label
                prev_key = key 
                segment_info.append((int(start*100), int(end*100)+1, label))
                

            feat_key = prev_key + "_{:06d}-{:06d}".format(int(start_time*100), int(prev_end_time*100)+1)
            output_dict[feat_key] = segment_info
    else:

        with open(vad_file, "r") as vads:
            for line in vads:


                if labels_given: 
                    key, start, end, label = line.strip(
                        ).split()
                else:
                    key, start, end = line.strip(
                        ).split()
                
                start = float(start)
                end = float(end)
                key = key.replace("_", "-")

                if labels_given and label in labels_to_exclude:
                    continue
                if prev_end_time != start or prev_key != key:
                    if prev_end_time != -1: 
                        start_frame = int(start_time*100)
                        prev_end_frame = int(prev_end_time*100)+1
                        
                        if prev_end_frame > feats[prev_key].shape[0]:
                            prev_end_frame = feats[prev_key].shape[0]
                        feat_key = prev_key + "_{:06d}-{:06d}".format(start_frame, prev_end_frame)
                        output_dict[feat_key] = segment_info
                    segment_info = []
                    start_time = start
                prev_end_time = end
                prev_key = key 
                segment_info.append((int(start*100), int(end*100)+1))
                

            feat_key = prev_key + "_{:06d}-{:06d}".format(int(start_time*100), int(prev_end_time*100)+1)
            output_dict[feat_key] = segment_info

    print("Number of segments: {}".format(len(output_dict)))

    return output_dict


    
def get_segments(feats, segments):

    print("\nExtracting segments:\n")

    output_feats = {}

    for key in sorted(segments):
            
        info = key.strip().split("_")
        feats_key = info[0]
        start = int(info[1].split("-")[0])
        end = int(info[1].split("-")[1])

        segement_key = feats_key + "_{:06d}-{:06d}".format(start, end)
        if end - start -1 != 0: 
            output_feats[segement_key] = feats[feats_key][start: end, :]
         
    print("Number of features: {}".format(len(output_feats)))  

    return output_feats
#_____________________________________________________________________________________________________________________________________
#
# Extracting subsets
#
#_____________________________________________________________________________________________________________________________________
    

def extract_subsets(lib, segments_or_words="Segments", **kwargs):

    dataset = lib["dataset"]
    out_dir = lib["out_dir"]
    feats_type = lib["feats_type"]

    print("\nExtracting subsets:\n")

    if segments_or_words=="Segments":
        feat_name = dataset + "_" + feats_type + "_segmented_features"
        list_name = dataset + "_" + feats_type + "_segments_list"

    util_library.check_dir(out_dir)
    
    segmented_subsets_path = path.join(out_dir, "Subsets", segments_or_words, feats_type)
    util_library.check_dir(segmented_subsets_path)
    
    in_dir = [path.join(out_dir, "Features", segments_or_words, feats_type, feat_name), path.join(out_dir, "Features", segments_or_words, feats_type, list_name)]
    seg_dir = [lib["training_speakers_path"], lib["validation_speakers_path"], lib["testing_speakers_path"]]
    out_seg_dir = [path.join(segmented_subsets_path, "seg_train_subset_" + feats_type), path.join(segmented_subsets_path, "seg_train_subset_list_" + feats_type), 
        path.join(segmented_subsets_path, "seg_val_subset_" + feats_type), path.join(segmented_subsets_path, "seg_val_subset_list_" + feats_type),
        path.join(segmented_subsets_path, "seg_test_subset_" + feats_type), path.join(segmented_subsets_path, "seg_test_subset_list_" + feats_type)]
    train_feats, train_list, val_feats, val_list, test_feats, test_list = subset_division(feats_type, in_dir, seg_dir, out_seg_dir)

    return train_feats, train_list, val_feats, val_list, test_feats, test_list
#_____________________________________________________________________________________________________________________________________
#
# Extracting ground truth words
#
#_____________________________________________________________________________________________________________________________________

def extract_words(feats, feats_list, lib, train_val_or_test, **kwargs):

    dataset = lib["dataset"]
    out_dir = lib["out_dir"]
    feats_type = lib["feats_type"]
    include_labels = lib["include_labels"]


    output_dict = {}

    if include_labels:

        for key in tqdm(sorted(feats_list)):
            basekey = key.strip().split("_")[0]
            base_start = int(key.strip().split("_")[1].split("-")[0])
            for (start, end, label) in feats_list[key]:

                start = int(start)
                end = int(end)
                new_key = "{}".format(label) + "_" + basekey + "_{:06d}-{:06d}".format(start, end)
                if end - start - 1 != 0:
                    output_dict[new_key] = feats[key][start-base_start:end-base_start, :]

    else:

        for key in tqdm(sorted(feats_list)):
            basekey = key.strip().split("_")[0]
            base_start = int(key.strip().split("_")[1].split("-")[0])
            for (start, end) in feats_list[key]:

                start = int(start)
                end = int(end)
                new_key = basekey + "_{:06d}-{:06d}".format(start, end)
                if end - start - 1 != 0:
                    output_dict[new_key] = feats[key][start-base_start:end-base_start, :]
        
    segmented_subsets_path = path.join(out_dir, "Subsets", "Words", feats_type)
    util_library.check_dir(segmented_subsets_path)
    save_dir = path.join(segmented_subsets_path, "gt_" + train_val_or_test + "_" + feats_type)
    print("Number of words in subset: {}".format(len(output_dict)))
    write_feats(output_dict, save_dir)

#_____________________________________________________________________________________________________________________________________
#
# Data segmentation into subsets
#
#_____________________________________________________________________________________________________________________________________

def write_subset(input_data, list_data, segments_path, output_path, list_output_path):

    segments = set([line.strip().split(" ")[0] for line in open(segments_path)])

    output_dict = {}
    list_dict = {}
    num_entries = 0
    
    for key in input_data:
        if key.startswith(tuple(segments)):
            output_dict[key] = input_data[key]
            list_dict[key] = list_data[key]
            num_entries += 1

    print("Number of features in subset:", num_entries)
    write_feats(output_dict, output_path)
    write_feats(list_dict, list_output_path)

    return output_dict, list_dict


def subset_division(feats_type, in_dir, seg_dir, out_dir):


    input_npz_path = path.join(in_dir[0] + ".npz")
    input_data = np.load(input_npz_path)

    list_npz_path = path.join(in_dir[1] + ".npz")
    list_data = np.load(list_npz_path)

    train_segements_path  = seg_dir[0]
    val_segements_path  = seg_dir[1]
    test_segements_path  = seg_dir[2]
    train_output_path = out_dir[0]
    train_list_path = out_dir[1]
    val_output_path = out_dir[2]
    val_list_path = out_dir[3]
    test_output_path = out_dir[4]
    test_list_path = out_dir[5]

    train, train_list = write_subset(input_data, list_data, train_segements_path, train_output_path, train_list_path)
    val, val_list = write_subset(input_data, list_data, val_segements_path, val_output_path, val_list_path)
    test, test_list = write_subset(input_data, list_data, test_segements_path, test_output_path, test_list_path)

    return train, train_list, val, val_list, test, test_list
