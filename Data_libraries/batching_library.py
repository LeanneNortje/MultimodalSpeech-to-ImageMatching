#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: Herman Kamper
#_________________________________________________________________________________________________
#
# This script contains various speech and image data iterators used throughout this repository.
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
from sklearn.neighbors import NearestNeighbors
import subprocess
import sys
from tqdm import tqdm
from scipy.fftpack import dct

sys.path.append("..")
from paths import data_path
from paths import general_lib_path
data_path = path.join("..", data_path)

sys.path.append(path.join("..", general_lib_path))
import util_library

#_____________________________________________________________________________________________________________________________________
#
# Speech data iterators with variable lengths
#
#_____________________________________________________________________________________________________________________________________


class speech_iterator(object):
    def __init__(self, input_list, batch_size, shuffle_batches_every_epoch=False, lengths=True, target_lengths=False):
        self.input_list = input_list
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.input_dim = self.input_list[0].shape[-1]
        self.input_lengths = np.array([k.shape[0] for k in self.input_list])
        self.num_batches = int(len(input_list)/batch_size)
        self.indices = np.arange(len(self.input_list))
        self.lengths = lengths
        self.target_lengths = target_lengths
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_indices = self.indices[i*self.batch_size: (i+1)*self.batch_size]

            this_batches_lengths = self.input_lengths[this_batches_indices]

            padded_inputs = np.zeros(
                (len(this_batches_indices), np.max(this_batches_lengths), self.input_dim),
                dtype=np.float32
                )

            for n, length in enumerate(this_batches_lengths):
                padded_inputs[n, :length, :] = self.input_list[this_batches_indices[n]]

            if self.target_lengths and self.lengths: yield (padded_inputs, this_batches_lengths, this_batches_lengths)
            elif self.lengths: yield (padded_inputs, this_batches_lengths)
            elif self.target_lengths==False and self.lengths==False: yield (padded_inputs) 

class pair_speech_iterator(object):
    def __init__(self, input_list, pair_list, batch_size, shuffle_batches_every_epoch=False,
        mask_x=None, return_mask=False):
        
        self.input_list = input_list
        self.pair_list = pair_list
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.mask_x = mask_x
        self.return_mask = return_mask

        self.input_dim = self.input_list[0].shape[-1]
        self.input_lengths = np.array([k.shape[0] for k in self.input_list])
        self.num_batches = int(len(self.pair_list)/self.batch_size)

        self.indices = np.arange(len(self.pair_list))
        
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_pair_list = [
                self.pair_list[n] for n in self.indices[
                i*self.batch_size: (i+1)*self.batch_size]
                ]

            first_pair_indices = [i for i, j in this_batches_pair_list]
            second_pair_indices = [j for i, j in this_batches_pair_list]

            first_pair_lengths = self.input_lengths[first_pair_indices]
            second_pair_lengths = self.input_lengths[second_pair_indices]

            num_to_pad = max(np.max(first_pair_lengths), np.max(second_pair_lengths))

            first_pair_padded_inputs = np.zeros(
                (len(first_pair_indices), num_to_pad, self.input_dim), dtype=np.float32
                )
            second_pair_padded_inputs = np.zeros(
                (len(second_pair_indices), num_to_pad, self.input_dim), dtype=np.float32
                )

            first_mask = np.zeros(
                (len(first_pair_indices), num_to_pad), dtype=np.float32
                )
            second_mask = np.zeros(
                (len(second_pair_indices), num_to_pad), dtype=np.float32
                )

            for n, length in enumerate(first_pair_lengths):
                first_pair_padded_inputs[n, :length, :] = self.input_list[
                    first_pair_indices[n]
                    ]
                first_mask[n, :length] = 1
            for n, length in enumerate(second_pair_lengths):
                second_pair_padded_inputs[n, :length, :] = self.input_list[
                    second_pair_indices[n]
                    ]
                second_mask[n, :length] = 1
                    
            if self.return_mask: 
                yield (first_pair_padded_inputs, first_pair_lengths, first_mask, 
                    second_pair_padded_inputs, second_pair_lengths, second_mask)
            else: 
                yield (first_pair_padded_inputs, first_pair_lengths, second_pair_padded_inputs, 
                second_pair_lengths)


class bucketing_speech_iterator(object):
    def __init__(self, input_list, batch_size, n_buckets, shuffle_batches_every_epoch=False):
        self.input_list = input_list
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch

        self.input_dim = self.input_list[0].shape[-1]
        self.input_lengths = np.array([k.shape[0] for k in self.input_list])
        self.num_batches = int(len(input_list)/batch_size)

        self.n_buckets = n_buckets
        sorted_indices = np.argsort(self.input_lengths)
        self.bucket_size = int(len(self.input_list)/self.n_buckets)
        self.buckets = []
        for n in range(self.n_buckets):
            self.buckets.append(
                sorted_indices[n*self.bucket_size:(n+1)*self.bucket_size]
                )
        self.shuffle()

    def shuffle(self):
        for n in range(self.n_buckets):
            np.random.shuffle(self.buckets[n])
        self.indices = np.concatenate(self.buckets)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_indices = self.indices[i*self.batch_size: (i+1)*self.batch_size]

            this_batches_lengths = self.input_lengths[this_batches_indices]

            padded_inputs = np.zeros(
                (len(this_batches_indices), np.max(this_batches_lengths), self.input_dim),
                dtype=np.float32
                )

            for n, length in enumerate(this_batches_lengths):
                padded_inputs[n, :length, :] = self.input_list[this_batches_indices[n]]

            yield (padded_inputs, this_batches_lengths)

class bucketing_pair_speech_iterator(object):
    def __init__(self, input_list, pair_list, batch_size, n_buckets, 
        shuffle_batches_every_epoch=False, mask_x=None, return_mask=False):

        self.input_list = input_list
        self.pair_list = pair_list
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.mask_x = mask_x
        self.return_mask = return_mask

        self.input_dim = self.input_list[0].shape[-1]
        self.input_lengths = np.array([k.shape[0] for k in self.input_list])
        self.num_batches = int(len(self.pair_list)/self.batch_size)

        self.n_buckets = n_buckets
        sorted_indices = np.argsort(
            [max(self.input_lengths[i], self.input_lengths[j]) for i, j in self.pair_list]
            )
        self.bucket_size = int(len(self.pair_list)/self.n_buckets)
        self.buckets = []
        for n in range(self.n_buckets):
            self.buckets.append(
                sorted_indices[n*self.bucket_size:(n+1)*self.bucket_size]
                )
        self.shuffle()

    def shuffle(self):
        for n in range(self.n_buckets):
            np.random.shuffle(self.buckets[n])
        self.indices = np.concatenate(self.buckets)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_pair_list = [
                self.pair_list[n] for n in self.indices[
                i*self.batch_size: (i+1)*self.batch_size]
                ]

            first_pair_indices = [i for i, j in this_batches_pair_list]
            second_pair_indices = [j for i, j in this_batches_pair_list]

            first_pair_lengths = self.input_lengths[first_pair_indices]
            second_pair_lengths = self.input_lengths[second_pair_indices]
            
            num_to_pad = max(np.max(first_pair_lengths), np.max(second_pair_lengths))

            first_pair_padded_inputs = np.zeros(
                (len(first_pair_indices), num_to_pad, self.input_dim), dtype=np.float32
                )
            second_pair_padded_inputs = np.zeros(
                (len(second_pair_indices), num_to_pad, self.input_dim), dtype=np.float32
                )
            first_mask = np.zeros(
                (len(first_pair_indices), num_to_pad), dtype=np.float32
                )
            second_mask = np.zeros(
                (len(second_pair_indices), num_to_pad), dtype=np.float32
                )

            for n, length in enumerate(first_pair_lengths):
                first_pair_padded_inputs[n, :length, :] = self.input_list[
                    first_pair_indices[n]
                    ]
                first_mask[n, :length] = 1

            for n, length in enumerate(second_pair_lengths):
                second_pair_padded_inputs[n, :length, :] = self.input_list[
                    second_pair_indices[n]
                    ]
                second_mask[n, :length] = 1
                
            if self.return_mask: 
                yield (first_pair_padded_inputs, first_pair_lengths, first_mask, 
                    second_pair_padded_inputs, second_pair_lengths, second_mask)
            else: 
                yield (first_pair_padded_inputs, first_pair_lengths, second_pair_padded_inputs, 
                    second_pair_lengths)


class speech_iterator_array(object):
    def __init__(self, input_x, batch_size, shuffle_batches_every_epoch=False, mask_x=None, return_mask=False):
        
        self.input_x = input_x
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.mask_x = mask_x
        self.return_mask = return_mask

        self.num_batches = int(len(self.input_x)/self.batch_size)

        self.indices = np.arange(len(self.input_x))
        
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_indices = [
                self.indices[i*self.batch_size: (i+1)*self.batch_size]
                ]

            if self.return_mask:
                yield (self.input_x[this_batches_indices], self.mask_x[this_batches_indices])

            else:
                yield (self.input_x[this_batches_indices])
            

class pair_speech_iterator_array(object):
    def __init__(self, input_x, pair_list, batch_size, shuffle_batches_every_epoch=False, mask_x=None, return_mask=False):
        
        self.input_x = input_x
        self.pair_list = pair_list
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.mask_x = mask_x
        self.return_mask = return_mask

        self.num_batches = int(len(self.pair_list)/self.batch_size)

        self.indices = np.arange(len(self.pair_list))
        
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_pair_list = [
                self.pair_list[n] for n in self.indices[
                i*self.batch_size: (i+1)*self.batch_size]
                ]

            first_pair_indices = [i for i, j in this_batches_pair_list]
            second_pair_indices = [j for i, j in this_batches_pair_list]

            if self.return_mask: yield (self.input_x[first_pair_indices], self.mask_x[first_pair_indices], self.input_x[second_pair_indices], self.mask_x[second_pair_indices])
            else: yield (self.input_x[first_pair_indices], self.input_x[second_pair_indices])


class speech_iterator_with_labels(object):
    def __init__(self, input_list, labels, batch_size, num_classes, shuffle_batches_every_epoch=False, return_labels=True):

        self.input_list = input_list
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.return_labels = return_labels
        self.input_dim = self.input_list[0].shape[-1]
        self.input_lengths = np.array([k.shape[0] for k in self.input_list])
        self.num_batches = int(len(input_list)/batch_size)
        self.indices = np.arange(len(self.input_list))
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_indices = self.indices[i*self.batch_size: (i+1)*self.batch_size]
            
            this_batches_lengths = self.input_lengths[this_batches_indices]

            batch_inputs = np.zeros(
                (len(this_batches_indices), np.max(this_batches_lengths), self.input_dim),
                dtype=np.float32
                )
            batch_labels = np.zeros(
                (len(this_batches_indices), self.num_classes),
                dtype=np.float32
                )
            for n, length in enumerate(this_batches_lengths):
                
                batch_inputs[n, :length, :] = self.input_list[this_batches_indices[n]]
                if self.return_labels:
                    batch_labels[n, self.labels[this_batches_indices[n]]] = 1

            if self.return_labels: yield (batch_inputs, this_batches_lengths, batch_labels)
            else: yield (batch_inputs, this_batches_lengths)

class speech_iterator_with_one_dimensional_labels(object):
    def __init__(self, input_list, pair_list, labels, batch_size, n_buckets, shuffle_batches_every_epoch=False, return_labels=True):

        self.input_list = input_list
        self.pair_list = pair_list
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.return_labels = return_labels
        self.n_buckets = n_buckets
        self.input_dim = self.input_list[0].shape[-1]
        self.input_lengths = np.array([k.shape[0] for k in self.input_list])
        self.num_batches = int(len(self.pair_list)/batch_size)
        
        sorted_indices = np.argsort(
            [max(self.input_lengths[i], self.input_lengths[j]) for i, j in self.pair_list]
            )
        self.bucket_size = int(len(self.pair_list)/self.n_buckets)
        self.buckets = []
        for n in range(self.n_buckets):
            self.buckets.append(
                sorted_indices[n*self.bucket_size:(n+1)*self.bucket_size]
                )
        self.shuffle()

    def shuffle(self):
        for n in range(self.n_buckets):
            np.random.shuffle(self.buckets[n])
        self.indices = np.concatenate(self.buckets)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_pair_list = [
                self.pair_list[n] for n in self.indices[
                i*self.batch_size: (i+1)*self.batch_size]
                ]

            first_pair_indices = [i for i, j in this_batches_pair_list]
            second_pair_indices = [j for i, j in this_batches_pair_list]
            this_batches_indices = list(
                set(first_pair_indices).union(set(second_pair_indices))
                )
            
            this_batches_lengths = self.input_lengths[this_batches_indices]

            batch_inputs = np.zeros(
                (len(this_batches_indices), np.max(this_batches_lengths), self.input_dim),
                dtype=np.float32
                )
            batch_labels = np.zeros(
                (len(this_batches_indices)),
                dtype=np.float32
                )

            for n, length in enumerate(this_batches_lengths):
                
                batch_inputs[n, :length, :] = self.input_list[this_batches_indices[n]]
                if self.return_labels:
                    batch_labels[n] = self.labels[this_batches_indices[n]]

            if self.return_labels: yield (batch_inputs, this_batches_lengths, batch_labels)
            else: yield (batch_inputs, this_batches_lengths)

#_____________________________________________________________________________________________________________________________________
#
# Image data iterators
#
#_____________________________________________________________________________________________________________________________________

class image_iterator(object):
    def __init__(self, input_list, batch_size, shuffle_batches_every_epoch=False):
        self.input_list = input_list
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.num_batches = int(len(input_list)/batch_size)
        self.indices = np.arange(len(self.input_list))
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_indices = self.indices[i*self.batch_size: (i+1)*self.batch_size]

            batch_inputs = np.zeros(
                (len(this_batches_indices), self.input_list[0].shape[-1]),
                dtype=np.float32
                )
            for n in range(len(this_batches_indices)):
                batch_inputs[n, :] = self.input_list[this_batches_indices[n]]

            yield (batch_inputs)

class pair_image_iterator(object):

    def __init__(self, input_list, pair_list, batch_size, shuffle_batches_every_epoch=False):

        self.input_list = input_list
        self.pair_list = pair_list
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch

        self.num_batches = int(len(pair_list)/batch_size)
        self.indices = np.arange(len(self.pair_list))

        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()
       
        for i_th_batch in range(self.num_batches):

            batch_pair_list = [
                self.pair_list[i] for i in self.indices[
                i_th_batch*self.batch_size:(i_th_batch + 1)*self.batch_size]
                ]

            this_batches_indices = [i for i, j in batch_pair_list]
            pair_batch_indices = [j for i, j in batch_pair_list]

            output_batch = np.zeros(
                (len(this_batches_indices), self.input_list[0].shape[-1]),
                dtype=np.float32
                )
            pair_batch = np.zeros(
                (len(this_batches_indices), self.input_list[0].shape[-1]),
                dtype=np.float32
                )

            for n in range(len(this_batches_indices)):
                output_batch[n, :] = self.input_list[this_batches_indices[n]]

            for n in range(len(pair_batch_indices)):
                pair_batch[n, :] = self.input_list[pair_batch_indices[n]]

            yield (output_batch, pair_batch)

class image_iterator_with_labels(object):
    def __init__(self, input_list, labels, batch_size, num_classes, shuffle_batches_every_epoch=False, return_labels=True):

        self.input_list = input_list
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.return_labels = return_labels
        self.num_batches = int(len(input_list)/batch_size)
        self.indices = np.arange(len(self.input_list))
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i in range(self.num_batches):

            this_batches_indices = self.indices[i*self.batch_size: (i+1)*self.batch_size]

            batch_inputs = np.zeros(
                (len(this_batches_indices), self.input_list[0].shape[-1]),
                dtype=np.float32
                )
            batch_labels = np.zeros(
                (len(this_batches_indices), self.num_classes),
                dtype=np.float32
                )
            for n in range(len(this_batches_indices)):
                
                batch_inputs[n, :] = self.input_list[this_batches_indices[n]]
                if self.return_labels:
                    batch_labels[n, self.labels[this_batches_indices[n]]] = 1
                    

            if self.return_labels: yield (batch_inputs, batch_labels)
            else: yield (batch_inputs)

class image_iterator_with_one_dimensional_labels(object):
    def __init__(self, input_list, pair_list, labels, batch_size, shuffle_batches_every_epoch=False, return_labels=True):

        self.input_list = input_list
        self.pair_list = pair_list
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle_batches_every_epoch = shuffle_batches_every_epoch
        self.return_labels = return_labels
        self.num_batches = int(len(pair_list)/batch_size)
        self.indices = np.arange(len(self.pair_list))
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):

        if self.shuffle_batches_every_epoch: self.shuffle()

        for i_th_batch in range(self.num_batches):

            batch_pair_list = [
                self.pair_list[i] for i in self.indices[
                i_th_batch*self.batch_size:(i_th_batch + 1)*self.batch_size]
                ]

            pair_1_batches_indices = [i for i, j in batch_pair_list]
            pair_2_batch_indices = [j for i, j in batch_pair_list]
            this_batches_indices = list(
                set(pair_1_batches_indices).union(set(pair_2_batch_indices))
                )

            batch_inputs = np.zeros(
                (len(this_batches_indices), self.input_list[0].shape[-1]),
                dtype=np.float32
                )
            batch_labels = np.zeros(
                (len(this_batches_indices)),
                dtype=np.float32
                )
            
            for n in range(len(this_batches_indices)):
                
                batch_inputs[n, :] = self.input_list[this_batches_indices[n]]
                if self.return_labels:
                    batch_labels[n] = self.labels[this_batches_indices[n]]
                    

            if self.return_labels: yield (batch_inputs, batch_labels)
            else: yield (batch_inputs)