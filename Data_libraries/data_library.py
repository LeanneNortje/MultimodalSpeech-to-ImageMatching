#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: Herman Kamper
#_________________________________________________________________________________________________
#
# This script contains various functions to read in data from .npz files, to generate pairs, to 
# read in pairs from text files and truncate data.

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
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import subprocess
import sys
from tqdm import tqdm
from scipy.fftpack import dct
import random

sys.path.append("..")
from paths import model_lib_path

sys.path.append(path.join("..", model_lib_path))
import model_setup_library

COL_LENGTH = model_setup_library.COL_LENGTH

#_____________________________________________________________________________________________________________________________________
#
# Data pairs
#
#_____________________________________________________________________________________________________________________________________

def data_pairs_from_file(file_fn, keys, add_both_directions=False):

    pair_list = []
    pair_set= set()
    pair_list = []

    for line in tqdm(open(file_fn, 'r'), desc="\tReading in pairs", ncols=COL_LENGTH):

        pairs = line.split()
        cur_key = np.where(np.asarray(keys) == pairs[0])[0][0]

        pair_key = np.where(np.asarray(keys) == pairs[1])[0][0]
        pair_set.add((cur_key, pair_key))
        if add_both_directions:
            pair_set.add((pair_key, cur_key))  
    

    for pair1, pair2 in pair_set:
        pair_list.append((pair1, pair2))

    return pair_list

def image_data_pairs(image_x, keys, labels):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(len(labels)), desc="\tGenerating pairs", ncols=COL_LENGTH):
        cur_label = labels[i]

        matching_labels = np.where(np.asarray(labels)== labels[i])[0]
        
        if len(matching_labels) > 0:
            count = 0
            pair = -1
            while pair == -1 and count < len(matching_labels):
                lower_limit = 0.05
                upper_limit = 0.25
                this_distance = cdist(image_x[i], image_x[matching_labels[count]], "cosine")

                if keys[i] != keys[matching_labels[count]] and this_distance > lower_limit and this_distance < upper_limit:
                    pair = matching_labels[count]
                count += 1
            
        if pair != -1: pair_list.append((i, pair))

    return pair_list

def data_pairs_from_different_speakers(labels, keys, add_both_directions=False):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(len(labels)), desc="\tGenerating pairs from different speakers", ncols=COL_LENGTH):

        cur_label = labels[i]
        speaker1 = keys[i].split("_")[1].split("-")[0]
        matching_labels = np.where(np.asarray(labels)== labels[i])[0]
        
        if len(matching_labels) > 0:
            count = 0
            pair = -1
            
            while pair == -1 and count < len(matching_labels):
                speaker2 = keys[matching_labels[count]].split("_")[1].split("-")[0]

                if speaker1 != speaker2:
                    pair = matching_labels[count] 
                    break
                count += 1
            if pair != -1: pair_list.append((i, pair))
                       
    return pair_list

def speech_data_pairs_from_file(file_fn, keys, add_both_directions=False):

    pair_list = []
    pair_set= set()
    pair_list = []

    for line in tqdm(open(file_fn, 'r'), desc="\tReading in pairs", ncols=COL_LENGTH):

        pairs = line.split()
        cur_key = np.where(np.asarray(keys) == pairs[0])[0][0]

        same_speaker_pair_key = np.where(np.asarray(keys) == pairs[1])[0][0]
        pair_set.add((cur_key, same_speaker_pair_key))
        if add_both_directions:
            pair_set.add((same_speaker_pair_key, cur_key))  

        different_speaker_pair_key = np.where(np.asarray(keys) == pairs[2])[0][0]
        pair_set.add((cur_key, different_speaker_pair_key))
        if add_both_directions:
            pair_set.add((different_speaker_pair_key, cur_key)) 
    

    for pair1, pair2 in pair_set:
        pair_list.append((pair1, pair2))

    return pair_list

def data_pairs(labels, add_both_directions=False):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(N-1), desc="\tGenerating pairs", ncols=COL_LENGTH):
        offset = i + 1
        cur_label = labels[i]
        matching_labels = np.where(np.asarray(labels[i + 1:])== labels[i])[0] + offset
        if len(matching_labels) > 0:
            pair_list.append((i, matching_labels[0]))
            if add_both_directions:
                pair_list.append((matching_labels[0], i))  

    return pair_list

def data_pairs_from_different_speakers(labels, keys, add_both_directions=False):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(N-1), desc="\tGenerating pairs from different speakers", ncols=COL_LENGTH):
        offset = i + 1
        cur_label = labels[i]
        speaker1 = keys[i].split("_")[1].split("-")[0]
        matching_labels = np.where(np.asarray(labels[i + 1:])== labels[i])[0] + offset
        if len(matching_labels) > 0:

            for j in range(len(matching_labels)):
                speaker2 = keys[matching_labels[j]].split("_")[1].split("-")[0]

                if speaker1 != speaker2:
                    pair_list.append((i, matching_labels[j]))
                    if add_both_directions:
                        pair_list.append((matching_labels[j], i))   
                    break
                       
    return pair_list

def data_pairs_from_same_and_different_speakers(labels, keys, add_both_directions=False):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(N-1), desc="\tGenerating pairs from different speakers", ncols=COL_LENGTH):
        offset = i + 1
        cur_label = labels[i]
        speaker1 = keys[i].split("_")[1].split("-")[0]
        matching_labels = np.where(np.asarray(labels[i + 1:])== labels[i])[0] + offset
        if len(matching_labels) > 0:

            for j in range(len(matching_labels)):
                speaker2 = keys[matching_labels[j]].split("_")[1].split("-")[0]

                if speaker1 != speaker2:
                    pair_list.append((i, matching_labels[j]))
                    if add_both_directions:
                        pair_list.append((matching_labels[j], i))   
                    break
                       
    return pair_list

def all_data_pairs(labels, add_both_directions=True):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(N-1), desc="\tGenrating all possible pairs", ncols=COL_LENGTH):
        offset = i + 1
        for matching_label_i in (np.where(np.asarray(labels[i + 1:]) == labels[i])[0] + offset):
            pair_list.append((i, matching_label_i))
            if add_both_directions:
                pair_list.append((matching_label_i, i))         
    return pair_list

#_____________________________________________________________________________________________________________________________________
#
# Loading in data
#
#_____________________________________________________________________________________________________________________________________

def load_image_data_from_npz(fn):
    npz = np.load(fn)

    feats = []
    words = []
    keys = []
    n_items = 0
    for im_key in tqdm(sorted(npz), desc="\tExtracting image data from {}".format(fn), ncols=COL_LENGTH):
        keys.append(im_key)
        feats.append(npz[im_key])
        word = im_key.split("_")[0]
        words.append(word)
        n_items += 1
        
    print("\tNumber of items extracted: {}".format(n_items))
    print("\tExample label of a feature: {}".format(words[0]))
    print("\tExample shape of a feature: {}".format(feats[0].shape))
    return (feats, words, keys)

def load_speech_data_from_npz(fn):
    npz = np.load(fn)

    feats = []
    words = []
    lengths = []
    keys = []
    n_items = 0
    for utt_key in tqdm(sorted(npz), desc="\tExtracting speech data from {}".format(fn), ncols=COL_LENGTH):
        keys.append(utt_key)
        feats.append(npz[utt_key])
        word = utt_key.split("_")[0]
        words.append(word)
        lengths.append(npz[utt_key].shape[0])
        n_items += 1
        
    print("\tNumber of items extracted: {}".format(n_items))
    print("\tExample label of a feature: {}".format(words[0]))
    print("\tExample shape of a feature: {}".format(feats[0].shape))
    return (feats, words, lengths, keys)

def load_latent_data_from_npz(fn):
    npz = np.load(fn)

    feats = []
    keys = []
    n_items = 0
    for key in tqdm(sorted(npz), desc="\tExtracting latents from {}".format(fn), ncols=COL_LENGTH):
        keys.append(key)
        feats.append(npz[key])
        n_items += 1
        
    print("\tNumber of items extracted: {}".format(n_items))
    print("\tExample key of a feature: {}".format(keys[0]))
    print("\tExample shape of a feature: {}".format(feats[0].shape))
    return (feats, keys)

#_____________________________________________________________________________________________________________________________________
#
# Truncating and limiting speech data
#
#_____________________________________________________________________________________________________________________________________


def truncate_data_dim(feats, lengths, max_feat_dim, max_frames):
    print("\n\tLimiting dimensionality: {}".format(max_feat_dim))
    print("\tLimiting number of frames: {}".format(max_frames))
    for i in tqdm(range(len(feats)), desc="\tTruncating", ncols=COL_LENGTH):
        feats[i] = feats[i][:max_frames, :max_feat_dim]
        lengths[i] = min(lengths[i], max_frames)

    
#_____________________________________________________________________________________________________________________________________
#
# Zeropadding speech data
#
#_____________________________________________________________________________________________________________________________________

def pad_speech_data(input_x, pad_to_num, return_mask=False):

    padded_data = np.zeros((len(input_x), pad_to_num, input_x[0].shape[1]), dtype=np.float32)
    lengths = []
    if return_mask: mask = np.zeros((len(input_x), pad_to_num), dtype=np.float32)

    for i, data in tqdm(enumerate(input_x), desc="\tPadding", ncols=COL_LENGTH):

        data_length = data.shape[0]
        padding = int((pad_to_num - data_length)/2)
        
        if data_length <= pad_to_num:
            padded_data[i, padding:padding+data_length, :] = data
            lengths.append(min(data_length, pad_to_num))
            if return_mask: mask[i, padding:padding+data_length] = 1

        else: 
            data_length = min(data_length, pad_to_num)
            padded_data[i, :data_length, :] = data[-padding:-padding+pad_to_num]
            lengths.append(data_length)
            if return_mask: mask[i, :] = 1

    if return_mask: return (padded_data, lengths, mask)
    else: return (padded_data, lengths)

#_____________________________________________________________________________________________________________________________________
#
# Getting a mask for speech data
#
#_____________________________________________________________________________________________________________________________________

def get_mask(input_x, input_lengths, max_length=None):
    if max_length is None: mask = np.zeros((len(input_x), max(input_lengths)), dtype=np.float32)
    else: mask = np.zeros((len(input_x), max_length), dtype=np.float32)

    for i, data in tqdm(enumerate(input_x), desc="\tGenerating mask", ncols=COL_LENGTH):

        data_length = data.shape[0]
        
        if data_length <= max(input_lengths): mask[i, 0:data_length] = 1
        elif max_length is not None and data_length <= max_length: mask[i, 0:data_length] = 1
        else: mask[i, :] = 1

    return mask


def flatten_speech_data(input_x, input_dim):

    input_x = np.transpose(input_x, (0, 2, 1))
    input_x = input_x.reshape((-1, input_dim))
    return input_x


def remove_test_classes(x, labels, lengths, keys, lab_to_exclude):
    speech_x = []
    speech_labels = []
    speech_lengths = []
    speech_keys = []
    for i, label in tqdm(enumerate(labels), desc="\tRemoving test classes", ncols=COL_LENGTH):
        if label not in lab_to_exclude:
            speech_x.append(x[i])
            speech_labels.append(labels[i])
            speech_lengths.append(lengths[i])
            speech_keys.append(keys[i])
    return (speech_x.copy(), speech_labels.copy(), speech_lengths.copy(), speech_keys.copy())

def test_classes(labels, lab_to_exclude, label_type):

    test_labels_present = False
    for label in tqdm(labels, desc="\tTesting {} labels".format(label_type), ncols=COL_LENGTH):
        if label in lab_to_exclude:
            test_labels_present = True

    if test_labels_present:
        print("\tTest classes present in {} dataset".format(label_type))