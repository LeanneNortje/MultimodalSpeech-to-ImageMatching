#!/usr/bin/env python

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
# --architecture:                           the hidden layer structure to construct the architecture ("fc", "rnn") 
#                                           {fc=fully connected or FFNN}
# --activation:                             the name of the atcivation to use between layers ("relu", "sigmoid")
# --learning_rate:                          any value to scale each parameter update
# --epochs:                                 the number of epochs to train the model for 
# --batch_size:                             the size of the batches to dividing the dataset in each epoch
# --n_buckets:                              number of buckets to divide the dataset into according to the number of speech frames, not 
#                                           applicable to images
# --enc:                                    the encoder layers given in a format where each layer dimension is divided by "_", 
#                                           i.e. 200_300_400 means an encoder with layer 3 layers of size 100, 200 and 300 in that 
#                                           precise order
# --latent:                                 the size of the latent or feature rpresentation
# --latent_enc:                           some encoder layers to encode the latent, given in a format where each layer dimension is 
#                                           divided by "_", i.e. 200_300_400 means an encoder with layer 3 layers of size 100, 200 and 
#                                           300 in that precise order
# --latent_func:                            the hidden layer structure to construct the latent enc-decoder 
# --data_type:                              the dataset to train the model on ("buckeye", "TIDigits", "omniglot", "MNIST") {if you 
#                                           choose an image dataset, omniglot or MNIST, you can only choose the --architecture as "fc" 
#                                           and if you choose a speech dataset, buckeye or TIDigits, you can only choose the 
#                                           --architecture as "rnn"}
# --features_type:                          the type of features to train on ("fbank", "mfcc", "None") {"fbank" and "mfcc" is applicable
#                                            to speech features and "None" to images}
# --train_tag:                              the method in which spoken words for training are isolated ("gt", "None") {"gt" is applicable
#                                            to speech features and "None" to images}
# --val_and_test_tag:                       the method in which spoken words for validation and testing are isolated ("gt", "None") 
#                                           {"gt" is applicable to speech features and "None" to images}
# --train_model:                            ("True", "False") indicating whether the model should be trained {"False" will restore 
#                                           the model}
# --test_model:                             ("True", "False") indicating whether the model should be tested
# --shuffle_batches_every_epoch:            ("True", "False") indicating whether the data in each batch should be shuffled before being
#                                            sent to the model
# --divide_into_buckets:                    ("True", "False") indicating whether data should be divided into buckets according to the 
#                                           number of speech frames, not applicable to images
# --max_frames:                             the maximum number of frames to limit the speech features to, not applicable to images
# --pretrain:                               ("True", "False") indicating whether the model should be pretrained
# --pretraining_model:                      the model type that the model should be pretrained as ("ae", "cae") {you can pretrain a 
#                                           model as itself and you can only pretrain a cae as an ae or vice versa}
# --pretraining_epochs:                     the number of epochs to pretrain the model for
# --use_one_shot_as_val:                    ("True", "False") indicating whether the model make use of a one-shot task to get validation
#                                            losses for early stopping during training
# --use_one_shot_as_val_for_pretraining:    ("True", "False") indicating whether the model make use of a one-shot task to get validation 
#                                           losses for early stopping during pretraining
# --one_shot_not_few_shot: :                ("True", "False") indicating to use one-shot of few-shot
# --do_one_shot_test:                       ("True", "False") indicating whether the model should be tested on an unimodal one-shot 
#                                           classification task
# --use_best_model: :                       ("True", "False") indicating whether the best model found with early stopping should be used 
#                                           as the final trained model, "False" wil use the model produced at the last epoch
# --use_best_pretrained_model:              ("True", "False") indicating whether the best pretrained model found with early stopping should 
#                                           be used to train the model from, "False" wil use the pretrained model produced at the last epoch
# --M:                                      the number of classes or concepts in the support set
# --K:                                      the number of examples of each class or concept in the support set
# --Q:                                      the number of queries in an episode 
# --margin:                                 the hinge loss margin which is only applicable to the Siamese models
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
# --kmeans:                                 ("True", "False") indicating whether the image pairs generated with kmeans should be used, if 
#                                           "False" the pairs generated with cosine distance will be used
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
    		if lib["architecture"] == "rnn": speech_model_library.rnn_speech_siamese(lib)
    	else:
    		if lib["architecture"] == "rnn": speech_model_library.rnn_speech_encdec(lib)
            
    elif lib["training_on"] == "images": 
    	if lib["model_type"] == "classifier":
    		if lib["architecture"] == "fc": vision_model_library.fc_vision_classifier(lib)
    	elif lib["model_type"] == "siamese":
    		if lib["architecture"] == "fc": vision_model_library.fc_vision_siamese(lib)
    	else:
    		if lib["architecture"] == "fc": vision_model_library.fc_vision_encdec(lib)

if __name__ == "__main__":
    main()