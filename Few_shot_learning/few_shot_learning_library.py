#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script contains code blocks necessary for the unimodal speech or image classification tasks,
# as well as the multimodal speech-image matching task. 
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
from sklearn import manifold, preprocessing
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import subprocess
import sys
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.fftpack import dct

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
# Construct an unimodal or multimodal support set
#
#_____________________________________________________________________________________________________________________________________

def construct_few_shot_support_set_with_keys(sp_x=None, sp_labels=None, sp_keys=None, sp_lengths=None, im_x=None, im_labels=None, im_keys=None, num_to_sample=11, num_of_each_sample=5, do_switch=True):
    
    #______________________________________________________________________________________________
    # Speech part
    #______________________________________________________________________________________________

    include_speech = True if sp_x is not None and sp_labels is not None and sp_lengths is not None and sp_keys is not None else False 
    if include_speech:
        speech_dict = sample_multiple_keys(
            sp_x, sp_labels, sp_keys, lengths=sp_lengths, num_to_sample=num_to_sample, num_of_each_sample=num_of_each_sample, do_switch=do_switch
            )
        speech_labels = [lab for lab in speech_dict]  
    #______________________________________________________________________________________________
    # Image part
    #______________________________________________________________________________________________

    include_images = True if im_x is not None and im_labels is not None and im_keys is not None else False
    if include_images: 
        
        image_dict = sample_multiple_keys(
            im_x, im_labels, im_keys, lengths=None, num_to_sample=num_to_sample, num_of_each_sample=num_of_each_sample,  
            labels_wanted=speech_labels.copy() if include_speech is True else None, #keys_to_not_include=[], 
            do_switch=do_switch
            )
        
    #______________________________________________________________________________________________
    # Construct support set
    #______________________________________________________________________________________________

    support_set = {}
    if include_speech and include_images:
        for key in speech_dict:
            support_set[key] = {}
            support_set[key]["image_data"] = image_dict[key]["data"]
            support_set[key]["image_keys"] = image_dict[key]["keys"]
            support_set[key]["speech_data"] = speech_dict[key]["data"]
            support_set[key]["speech_keys"] = speech_dict[key]["keys"]
            support_set[key]["speech_lengths"] = speech_dict[key]["lengths"]
    elif include_speech:
        for key in speech_dict:
            support_set[key] = {}
            support_set[key]["speech_data"] = speech_dict[key]["data"]
            support_set[key]["speech_keys"] = speech_dict[key]["keys"]
            support_set[key]["speech_lengths"] = speech_dict[key]["lengths"]
    elif include_images:
        for key in image_dict:
            support_set[key] = {}
            support_set[key]["image_data"] = image_dict[key]["data"]
            support_set[key]["image_keys"] = image_dict[key]["keys"]

    return support_set 


#_____________________________________________________________________________________________________________________________________
#
# Sampling multiple specified keys
#
#_____________________________________________________________________________________________________________________________________

def sample_multiple_keys(x, labels, keys, lengths=None, num_to_sample=10, num_of_each_sample=1, labels_wanted=None, exclude_key_list=[], do_switch=False):
    
    data_dict = {}
    key_list = exclude_key_list.copy()


    if lengths is not None:
        for i in range(num_of_each_sample):
            x_data, x_labels, x_lengths, x_keys = sampling_with_keys(
                x, labels, keys, num_to_sample, labels_wanted=labels_wanted, lengths=lengths, do_switch=do_switch,
                exclude_keys=key_list
                )
            for i, label in enumerate(x_labels):
                if label not in data_dict: 
                    data_dict[label] = {}
                    data_dict[label]["data"] = []
                    data_dict[label]["lengths"] = []
                    data_dict[label]["keys"] = []
                if x_keys[i] in data_dict[label]["keys"]: 
                    print("Key already in list")
                    continue
                data_dict[label]["data"].append(x_data[i])
                data_dict[label]["lengths"].append(x_lengths[i])
                data_dict[label]["keys"].append(x_keys[i])
            key_list.extend(x_keys)
            labels_wanted = x_labels
            
    else:
        for i in range(num_of_each_sample):
            x_data, x_labels, x_keys = sampling_with_keys(
                x, labels, keys, num_to_sample, labels_wanted=labels_wanted, lengths=None, do_switch=do_switch,
                exclude_keys=key_list
                )
            for i, label in enumerate(x_labels):
                if label not in data_dict: 
                    data_dict[label] = {}
                    data_dict[label]["data"] = []
                    data_dict[label]["keys"] = []
                if x_keys[i] in data_dict[label]["keys"]: 
                    print("Key already in list")
                    continue
                data_dict[label]["data"].append(x_data[i])
                data_dict[label]["keys"].append(x_keys[i])
              
            key_list.extend(x_keys)
            labels_wanted = x_labels

    return data_dict

def sampling_with_keys(x, labels, keys, num_to_sample, labels_wanted=None, lengths=None, do_switch=True, exclude_keys=[]):
    np.random.seed(42)
    counter = 0
    support_set_indices = []
    support_set_x = []
    labels_to_get = (
        [] if labels_wanted is None else
        labels_wanted
        )
    support_set_labels = []
    support_set_lengths = []
    support_set_keys = []
    
    while(counter < num_to_sample):

        index = random.randint(0, len(x)-1)

        cur_label = str(labels[index])
        label_to_get = cur_label
        if cur_label == '0' and do_switch: 
            z_or_o = random.randint(0, 1)
            if z_or_o == 0:
                label_to_get = 'z' if 'z' not in support_set_labels else 'o'
            else: 
                label_to_get = 'o' if 'o' not in support_set_labels else 'z' 
            
        
        if labels_wanted is None:

            if label_to_get not in support_set_labels and index not in support_set_indices and keys[index] not in exclude_keys:
                support_set_indices.append(index)
                support_set_x.append(x[index])
                support_set_labels.append(label_to_get)
                support_set_keys.append(keys[index])
                if lengths is not None: support_set_lengths.append(lengths[index])
                counter += 1
        else:
            
            if label_to_get in labels_to_get and label_to_get not in support_set_labels and index not in support_set_indices and keys[index] not in exclude_keys:
                support_set_indices.append(index)
                support_set_x.append(x[index])
                support_set_labels.append(label_to_get)
                support_set_keys.append(keys[index])
                if lengths is not None: support_set_lengths.append(lengths[index])
                counter += 1

    if lengths is not None: 
        return support_set_x, support_set_labels, support_set_lengths, support_set_keys 

    else: 
        return support_set_x, support_set_labels, support_set_keys

#_____________________________________________________________________________________________________________________________________
#
# Generating label matches grids which are used in the unimodal and multimodal one- or few-shot tasks
#
#_____________________________________________________________________________________________________________________________________

def label_test(label1, label2):

    if label1 == "0" and label2 == "z": return True
    if label1 == "0" and label2 == "o": return True
    if label1 == "z" and label2 == "0": return True
    if label1 == "o" and label2 == "0": return True
    else: return label1 == label2

def label_matches_grid_generation(labels_fn, num_queries, M, K):
    
    label_grid = np.zeros((num_queries, M*K), dtype=np.bool)
    labels_file = open(labels_fn, 'r')
    for n, line in enumerate(labels_file):

        (label1, label2) = line.split()

        x = int(n/(M*K))
        y = int((n-(x*M*K))%(M*K))

        label_grid[x, y] = label_test(label1, label2)

    return label_grid

def label_matches_grid_generation_2D(query_labels, set_labels):
    
    label_grid = np.zeros((len(query_labels), len(set_labels)), dtype=np.bool)
    for i, q_lab in enumerate(query_labels):
        for j, S_lab in enumerate(set_labels):
            label_grid[i, j] = label_test(q_lab, S_lab)

    return label_grid

def key_grid_generation(keys_fn, num_queries, M, K):
    
    key_grid = np.empty((num_queries, M*K), dtype='|S100')
    key_file = open(keys_fn, 'r')

    for n, line in enumerate(key_file):

        (key1, key2) = line.split()
        x = int(n/(M*K))
        y = int((n-(x*M*K))%(M*K))
        key_grid[x, y] = key2

    return key_grid