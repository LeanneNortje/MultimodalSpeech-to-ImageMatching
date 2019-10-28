#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
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
import subprocess
import sys
from tqdm import tqdm
from scipy.fftpack import dct

#_____________________________________________________________________________________________________________________________________
#
# Data pairs
#
#_____________________________________________________________________________________________________________________________________

def data_pairs_from_file(file_fn, keys, add_both_directions=False):

    pair_list = []
    pair_set= set()
    pair_list = []

    for line in open(file_fn, 'r'):

        pairs = line.split()
        cur_key = np.where(np.asarray(keys) == pairs[0])[0][0]

        # for i in range(1, len(pairs)):
        pair_key = np.where(np.asarray(keys) == pairs[1])[0][0]
        pair_set.add((cur_key, pair_key))
        if add_both_directions:
            pair_set.add((pair_key, cur_key))  
    

    for pair1, pair2 in pair_set:
        pair_list.append((pair1, pair2))

    return pair_list

def data_pairs(labels, add_both_directions=False):

    N = len(labels)
    pair_list = []

    for i in range(N-1):
        offset = i + 1
        cur_label = labels[i]
        matching_labels = np.where(np.asarray(labels[i + 1:])== labels[i])[0] + offset
        if len(matching_labels) > 0:
            pair_list.append((i, matching_labels[0]))
            if add_both_directions:
                pair_list.append((matching_labels[0], i))      
    return pair_list

def all_data_pairs(labels, add_both_directions=True):

    N = len(labels)
    pair_list = []

    for i in range(N-1):
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
    print("Extracting: {}".format(fn))
    npz = np.load(fn)

    feats = []
    words = []
    keys = []
    n_items = 0
    for im_key in tqdm(sorted(npz)):
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
    print("Extracting: {}".format(fn))
    npz = np.load(fn)

    feats = []
    words = []
    lengths = []
    keys = []
    n_items = 0
    for utt_key in tqdm(sorted(npz)):
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

#_____________________________________________________________________________________________________________________________________
#
# Truncating and limiting speech data
#
#_____________________________________________________________________________________________________________________________________

def truncate_data_dim(feats, lengths, max_feat_dim, max_frames):
    for i in range(len(feats)):
        feats[i] = feats[i][:max_frames, :max_feat_dim]
        lengths[i] = min(lengths[i], max_frames)

    
#_____________________________________________________________________________________________________________________________________
#
# Getting a mask for speech data
#
#_____________________________________________________________________________________________________________________________________

def get_mask(input_x, input_lengths, max_length=None):

    if max_length is None: mask = np.zeros((len(input_x), max(input_lengths)), dtype=np.float32)
    else: mask = np.zeros((len(input_x), max_length), dtype=np.float32)

    for i, data in enumerate(input_x):

        data_length = data.shape[0]
        
        if data_length <= max(input_lengths): mask[i, 0:data_length] = 1
        elif max_length is not None and data_length <= max_length: mask[i, 0:data_length] = 1
        else: mask[i, :] = 1

    return mask