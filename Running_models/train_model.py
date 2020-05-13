#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script trains a model from a library.
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
from paths import model_lib_path

sys.path.append(path.join("..", model_lib_path))
import speech_model_library
import vision_model_library
import model_setup_library

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#_____________________________________________________________________________________________________________________________________
#
# The options available to give ./train_model, where there are fixed values to the option, it is given between () brackets in the 
# description of each below. Important NOTES are given between {} brackets. 
#  
# --model_type:                             the type of model ("ae", "cae", "classifier", "siamese")
# --architecture:                           the hidden layer structure to construct the architecture ("cnn", "rnn") 
# --final_model:                            ("True", "False") indicating whether this is a final model to be saved in a separate folder
# --data_type:                              the dataset to train the model on ("buckeye", "TIDigits", "omniglot", "MNIST") {if you 
#                                           choose an image dataset, omniglot or MNIST, you can only choose the --architecture as "cnn" 
#                                           and if you choose a speech dataset, buckeye or TIDigits, you can only choose the 
#                                           --architecture as "rnn"}
# --other_image_dataset:                    the image dataset to mix the the dataset specified in data_type with ("MNIST", "omniglot")
# --other_speech_dataset:                   the speech dataset to mix the the dataset specified in data_type with ("buckeye", "TIDigits")
# --features_type:                          the type of features to train on ("fbank", "mfcc", "None") {"fbank" and "mfcc" is applicable
# --train_tag:                              the method in which spoken words for training are isolated ("gt", "None") {"gt" is applicable
#                                           to speech features and "None" to images}
# --max_frames:                             the maximum number of frames to limit the speech features to, not applicable to images
# --mix_training_datasets:                  ("True", "False") indicating whether the datasets in data_type should be mixed with the
#                                           other dataset
# --train_model:                            ("True", "False") indicating whether the model should be trained {"False" will restore 
#                                           the model}
# --use_best_model:                         ("True", "False") indicating whether the best model found with early stopping should be used 
#                                           as the final trained model, "False" wil use the model produced at the last epoch
# --test_model:                             ("True", "False") indicating whether the model should be tested
# --activation:                             the name of the activation to use between layers ("relu", "sigmoid")
# --batch_size:                             the size of the batches to dividing the dataset in each epoch
# --n_buckets:                              number of buckets to divide the dataset into according to the number of speech frames, not 
#                                           applicable to images
# --margin:                                 the hinge loss margin which is only applicable to the Siamese models
# --sample_n_classes:                       the number of classes per Siamese batch to sample
# --sample_k_examples:                      the number of examples to sample fo each of the sample_n_classes classes
# --n_siamese_batches:                      the number of Siamese batches for each epoch
# --rnn_type:                               the type of rnn cell to use in rnn layers
# --epochs:                                 the number of epochs to train the model for 
# --learning_rate:                          any value to scale each parameter update
# --keep_prob:                              the keep probability to use for each layer
# --shuffle_batches_every_epoch:            ("True", "False") indicating whether the data in each batch should be shuffled before being
#                                            sent to the model
# --divide_into_buckets:                    ("True", "False") indicating whether data should be divided into buckets according to the 
#                                           number of speech frames, not applicable to images
# --one_shot_not_few_shot:                  ("True", "False") indicating to use one-shot of few-shot
# --do_one_shot_test:                       ("True", "False") indicating whether the model should be tested on an unimodal one-shot 
#                                           classification task
# --do_few_shot_test:                       ("True", "False") indicating whether the model should be tested on an unimodal few-shot 
#                                           classification task 
# --pair_type:                              ("siamese", "classifier", "default") the type of distance metric used to generate 
#                                           pairs, the default is "cosine"
# --overwrite_pairs:                        ("True", "False") indicating whether ground truth labels should be used for data that are used
#                                           as unlabelled
# --pretrain:                               ("True", "False") indicating whether the model should be pretrained
# --pretraining_model:                      the model type that the model should be pretrained as ("ae", "cae") {you can pretrain a 
#                                           model as itself and you can only pretrain a cae as an ae or vice versa}     
# --pretraining_data:                       the dataset to pretrain the model on ("buckeye", "TIDigits", "omniglot", "MNIST") {if you 
#                                           choose an image dataset, omniglot or MNIST, you can only choose the --architecture as "cnn" 
#                                           and if you choose a speech dataset, buckeye or TIDigits, you can only choose the 
#                                           --architecture as "rnn"}
# --pretraining_epochs:                     the number of epochs to pretrain the model for                    
# --other_pretraining_image_dataset:        the image dataset to mix the the pretraining dataset specified in pretraining_data with 
#                                           ("MNIST", "omniglot")
# --other_pretraining_speech_dataset:       the speech dataset to mix the the pretraining dataset specified in pretraining_data with 
#                                           ("buckeye", "TIDigits")
# --use_best_pretrained_model:              ("True", "False") indicating whether the best pretrained model found with early stopping should 
#                                           be used to train the model from, "False" wil use the pretrained model produced at the last epoch
# --M:                                      the number of classes or concepts in the support set
# --K:                                      the number of examples of each class or concept in the support set
# --Q:                                      the number of queries in an episode 
# --one_shot_batches_to_use:                the subset to use on a unimodal classification task for testing the model 
#                                           ("train", "validation", "test")
# --one_shot_image_dataset:                 the image dataset to use on a unimodal classification task for testing a model 
#                                           ("MNIST", "omniglot")
# --one_shot_speech_dataset:                the speech dataset to use on a unimodal classification task for testing a model 
#                                           ("TIDigits", "buckeye")
# --validation_image_dataset:               the image dataset to use on a unimodal classification task for validation during training of a 
#                                           model ("MNIST", "omniglot")
# --validation_speech_dataset:              the speech dataset to use on a unimodal classification task for validation during training of a 
#                                           model ("TIDigits", "buckeye")
# --test_on_one_shot_dataset:               ("True", "False") indicating whether the unimodal classification task should be done on the 
#                                           specified one_shot_image_dataset or one_shot_speech_dataset
# --validate_on_validation_dataset:         ("True", "False") indicating whether the unimodal classification validation task should be done on 
#                                           the specified validation_image_dataset or validation_speech_dataset
# --enc:                                    the encoder layers given in a format where each layer dimension is divided by "_", 
#                                           i.e. 200_300_400 means an encoder with layer 3 layers of size 100, 200 and 300 in that 
#                                           precise order
# --latent:                                 the size of the latent or feature rpresentation
# --latent_enc:                             some encoder layers to encode the latent, given in a format where each layer dimension is 
#                                           divided by "_", i.e. 200_300_400 means an encoder with layer 3 layers of size 100, 200 and 
#                                           300 in that precise order
# --latent_func:                            the hidden layer structure to construct the latent enc-decoder 
#                                            to speech features and "None" to images}
# --rnd_seed:                               the random seed used to initialize the random number generator
#
#_____________________________________________________________________________________________________________________________________


#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():

    lib = model_setup_library.model_library_setup()
    if lib["training_on"] == "speech": 
    	if lib["model_type"] == "classifier":
    		if lib["architecture"] == "rnn": speech_model_library.rnn_speech_classifier(lib)
    	elif lib["model_type"] == "siamese":
    		if lib["architecture"] == "rnn": speech_model_library.rnn_speech_siamese_model(lib)
    	else:
    		if lib["architecture"] == "rnn": speech_model_library.rnn_speech_model(lib)
    		elif lib["architecture"] == "cnn": speech_model_library.cnn_speech_model(lib)
    		elif lib["architecture"] == "fc": speech_model_library.fc_speech_model(lib)
    elif lib["training_on"] == "images": 
    	if lib["model_type"] == "classifier":
            if lib["architecture"] == "cnn": vision_model_library.cnn_vision_classifier_model(lib)
            elif lib["architecture"] == "fc": vision_model_library.fc_vision_classifier_model(lib)
    	elif lib["model_type"] == "siamese":
            if lib["architecture"] == "cnn": vision_model_library.cnn_vision_siamese_model(lib)
            elif lib["architecture"] == "fc": vision_model_library.fc_vision_siamese_model(lib)
    	else:
    		if lib["architecture"] == "cnn": vision_model_library.cnn_vision_model(lib)
    		elif lib["architecture"] == "fc": vision_model_library.fc_vision_model(lib)

if __name__ == "__main__":
    main()