#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: Herman Kamper
#_________________________________________________________________________________________________
#
# This script contains various building blocks to implement various models in tensorflow.
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
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import timeit
import subprocess
from tqdm import tqdm

#_____________________________________________________________________________________________________________________________________
#
# General functions 
#
#_____________________________________________________________________________________________________________________________________

def saving_best_model(val_loss, min_val_loss, session, directory, saver):

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        path = saver.save(session, directory)
        save = "Saved"
    else: 
        save = " - "
        path = "Did not save model"

    return save, path, min_val_loss

def saving_model(session, directory, saver):

    path = saver.save(session, directory)

    return path

def feeding_dict(placeholders, values):
    if len(placeholders) == 1: return{placeholders[0]: values}
    else: return {key: value for key, value in zip(placeholders, values)}


#_____________________________________________________________________________________________________________________________________
#
# Layer functions
#
#_____________________________________________________________________________________________________________________________________

def fully_connected_layer(input_x, output_dim, name, print_layer=True):

    with tf.variable_scope(name):
        input_dim = input_x.get_shape().as_list()[-1]
        weight = tf.get_variable(
            "weight", [input_dim, output_dim], tf.float32, 
            initializer=tf.contrib.layers.xavier_initializer()
            )
        bias = tf.get_variable(
            "bias", [output_dim], tf.float32, initializer=tf.random_normal_initializer()
            )
        next_layer = tf.add(tf.matmul(input_x, weight), bias)

        if print_layer: print("\tFully connected layer of size {} and name {}".format(next_layer.shape, name))

        return next_layer


def rnn_cell(output_dim, rnn_type, **kwargs):

    cell = tf.nn.rnn_cell.BasicRNNCell(output_dim, **kwargs)

    if rnn_type == "lstm":
        args_to_lstm = {"state_is_tuple": True}
        args_to_lstm.update(kwargs)
        cell = tf.nn.rnn_cell.LSTMCell(output_dim, **args_to_lstm)

    elif rnn_type == "gru":
        cell = tf.nn.rnn_cell.GRUCell(output_dim, **kwargs)

    return cell


def rnn_layer(input_x, input_lengths, output_dim, name, rnn_type="lstm", keep_prob=1.0, scope=None, print_layer=True, **kwargs):

    with tf.variable_scope(name):

        cell = rnn_cell(output_dim, rnn_type, **kwargs)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1., output_keep_prob=keep_prob)
        output, final_state = tf.nn.dynamic_rnn(cell, input_x, sequence_length=input_lengths, dtype=tf.float32, scope=scope)
        if print_layer: print("\tRNN layer of size {} and name {}".format(output.shape, name))

    return output, final_state

def multiple_rnn_layers(input_x, input_lengths, hidden_layers, lib, layer_offset, print_layer=True, **kwargs):

    for layer_num, layer_size in enumerate(hidden_layers):
        output, final_state = rnn_layer(input_x, input_lengths, layer_size, "rnn_layer_{}".format(layer_num+layer_offset), rnn_type=lib["rnn_type"], keep_prob=lib["keep_prob"], print_layer=print_layer, **kwargs)
        input_x = output

    return output, final_state

#_____________________________________________________________________________________________________________________________________
#
# Model architectures
#
#_____________________________________________________________________________________________________________________________________

def fc_architecture(input_x, enc_layers, latent_layer, dec_layers, lib, activation_lib, layer_offset=0, print_layer=True):
    activation = activation_lib[lib["activation"]]

    if layer_offset == 0: 
        input_x = tf.reshape(input_x, [-1, lib["input_dim"]])
        if lib["input_dim"] not in dec_layers: dec_layers.append(lib["input_dim"])

    layer_counter = layer_offset

    for layer_num, layer_size in enumerate(enc_layers):
        input_x = fully_connected_layer(input_x, layer_size, "layer_{}".format(layer_counter), print_layer)
        if layer_num != len(enc_layers) - 1: input_x = activation(input_x)
        layer_counter += 1

    
    input_x = fully_connected_layer(input_x, latent_layer, "layer_{}".format(layer_counter), print_layer)
    latent = input_x
    layer_counter += 1

    for layer_num, layer_size in enumerate(dec_layers):
        input_x = fully_connected_layer(input_x, layer_size, "layer_{}".format(layer_counter), print_layer)
        if layer_num != len(dec_layers) - 1: input_x = activation(input_x)
        layer_counter += 1

    output = input_x

    return {"latent": latent, "output": output}


def rnn_architecture(input_placeholders, activation_lib, lib, output_lengths=None, print_layer=True, **kwargs):

    input_x = input_placeholders[0]
    input_lengths = input_placeholders[1]
    max_input_length = (
        tf.reduce_max(input_lengths) if output_lengths is None else
        tf.reduce_max([tf.reduce_max(input_lengths), tf.reduce_max(output_lengths)])
        )
    input_dim = input_x.get_shape().as_list()[-1]
    layer_count = 0
    enc_output, enc_final_state = multiple_rnn_layers(input_x, input_lengths, lib["enc"], lib, 0, print_layer, **kwargs)
    layer_count += len(lib["enc"])
    activation = activation_lib[lib["activation"]]

    if lib["rnn_type"] == "lstm":
        c, h = enc_final_state
    else:
        c = enc_final_state

    if lib["latent_func"] != "default":
        if lib["latent_func"] == "fc": 
            if lib["dec"][0] not in lib["latent_dec"]: lib["latent_dec"].append(lib["dec"][0])
            func = fc_architecture
            latent_model = func(c, lib["latent_enc"], lib["latent"], lib["latent_dec"], lib, activation_lib, layer_count, print_layer)
            input_x = latent_model["output"]
            latent = latent_model["latent"]
            layer_count += len(lib["latent_enc"]) + 1 + len(lib["latent_dec"])

    else: 
        input_x = c 
        input_x = fully_connected_layer(input_x, lib["latent"], "layer_{}".format(layer_count), print_layer)
        latent = input_x
        layer_count += 1

        input_x = fully_connected_layer(input_x, lib["dec"][0], "layer_{}".format(layer_count), print_layer)
        input_x = activation(input_x)
        layer_count += 1
    
    layer_to_dec_dim = input_x.get_shape().as_list()[-1]
    dec_input = tf.reshape(tf.tile(input_x, [1, max_input_length]), [-1, max_input_length, layer_to_dec_dim])

    dec_output, dec_final_state = multiple_rnn_layers(
        dec_input, input_lengths if output_lengths is None else output_lengths, lib["dec"], 
        lib, layer_count, print_layer, **kwargs)
    layer_count += len(lib["dec"])

    dec_output = tf.reshape(dec_output, [-1, lib["dec"][-1]])
    dec_output = fully_connected_layer(dec_output, input_dim, "layer_{}".format(layer_count), print_layer)
    dec_output = tf.reshape(dec_output, [-1, max_input_length, input_dim])
    
    return {"latent": latent, "output": dec_output}

def fc_classifier_architecture(input_x, enc_layers, latent_layer, lib, activation_lib, print_layer=True):
    activation = activation_lib[lib["activation"]]

    input_x = tf.reshape(input_x, [-1, lib["input_dim"]])

    for layer_num, layer_size in enumerate(enc_layers):
        input_x = fully_connected_layer(input_x, layer_size, "layer_{}".format(layer_num), print_layer)
        if layer_num != len(enc_layers) - 1: input_x = activation(input_x)
    
    input_x = fully_connected_layer(input_x, latent_layer, "layer_{}".format(len(enc_layers)), print_layer)
    latent = input_x

    input_x = fully_connected_layer(input_x, lib["num_classes"], "layer_{}".format(len(enc_layers)+1), print_layer)
    output = input_x

    return {"latent": latent, "output": output}


def rnn_classifier_architecture(input_placeholders, activation_lib, lib, print_layer=True, **kwargs):

    input_x = input_placeholders[0]
    input_lengths = input_placeholders[1]
    
    enc_output, enc_final_state = multiple_rnn_layers(input_x, input_lengths, lib["enc"], lib, 0, print_layer, **kwargs)

    if lib["rnn_type"] == "lstm":
        c, h = enc_final_state
    else:
        c = enc_final_state

    if lib["latent_func"] != "default":
        if lib["latent_func"] == "fc": 
            if lib["dec"][0] not in lib["latent_dec"]: lib["latent_dec"].append(lib["dec"][0])
            func = fc_architecture
            latent_model = func(c, lib["latent_enc"], lib["latent"], [], lib, activation_lib, len(lib["enc"]), print_layer)
            input_x = latent_model["output"]
            latent = latent_model["latent"]

    else: 
        input_x = c 
        input_x = fully_connected_layer(input_x, lib["latent"], "layer_{}".format(len(lib["enc"])), print_layer)
        latent = input_x

    input_x = fully_connected_layer(input_x, lib["num_classes"], "layer_{}".format(len(lib["enc"])+1), print_layer)
    output = input_x
    
    return {"latent": latent, "output": output}

def siamese_fc_architecture(input_placeholders, lib, activation_lib, print_layer=True):
    activation = activation_lib[lib["activation"]]

    input_x = input_placeholders[0]
    input_x = tf.reshape(input_x, [-1, lib["input_dim"]])

    for layer_num, layer_size in enumerate(lib["enc"]):
        input_x = fully_connected_layer(input_x, layer_size, "layer_{}".format(layer_num), print_layer)
        if layer_num != len(lib["enc"]) - 1: input_x = activation(input_x)
    
    input_x = fully_connected_layer(input_x, lib["latent"], "layer_{}".format(len(lib["enc"])), print_layer)
    latent = input_x

    return {"output": latent}

def siamese_rnn_architecture(input_placeholders, activation_lib, lib, print_layer=True, **kwargs):

    input_x = input_placeholders[0]
    input_lengths = input_placeholders[1]
    
    enc_output, enc_final_state = multiple_rnn_layers(input_x, input_lengths, lib["enc"], lib, 0, print_layer, **kwargs)

    if lib["rnn_type"] == "lstm":
        input_x, h = enc_final_state
    else:
        input_x = enc_final_state

    if lib["latent_func"] != "default":
        if lib["latent_func"] == "fc": 
            if lib["dec"][0] not in lib["latent_dec"]: lib["latent_dec"].append(lib["dec"][0])
            func = fc_architecture
            latent_model = func(input_x, lib["latent_enc"], lib["latent"], [], lib, activation_lib, len(lib["enc"]), print_layer)
            input_x = latent_model["output"]
            latent = latent_model["latent"]

    else:  
        input_x = fully_connected_layer(input_x, lib["latent"], "layer_{}".format(len(lib["enc"])), print_layer)
        latent = input_x
    
    return {"output": latent}
#_____________________________________________________________________________________________________________________________________
#
# Model value retrieval
#
#_____________________________________________________________________________________________________________________________________

def model_values(states, input_placeholders, data_it, lib, model_fn, labels=None, keys=None):

    input_embeddings = {}
    output_embeddings = {}
    latent_embeddings = {}
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    i = 0

    print("Restoring {}-model values from {}...\n".format(lib["model_name"], model_fn))
    saver = tf.train.Saver()
    with tf.Session(config=config) as sesh:
        saver.restore(sesh, model_fn)
        for values in data_it:
            feed_dict = feeding_dict(input_placeholders, values)

            outputs = sesh.run([states[0]], feed_dict)[0]
            latents = sesh.run([states[1]], feed_dict)[0]

            feats = values[0]
            if len(input_placeholders)==1: feats = values

            input_embeddings["embeddings"] = np.array(feats)
            if labels is not None: input_embeddings["labels"] = np.array(labels)
            if keys is not None: input_embeddings["keys"] = np.array(keys)
            output_embeddings["embeddings"] = np.array(outputs)
            if labels is not None: output_embeddings["labels"] = input_embeddings["labels"]
            if keys is not None: output_embeddings["keys"] = input_embeddings["keys"]
            latent_embeddings["embeddings"] = np.array(latents)
            if labels is not None: latent_embeddings["labels"] = input_embeddings["labels"]
            if keys is not None: latent_embeddings["keys"] = input_embeddings["keys"]

    return input_embeddings, output_embeddings, latent_embeddings