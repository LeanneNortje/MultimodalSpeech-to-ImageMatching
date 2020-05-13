#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script contains the contruction of separate speech and image models aswell as their basic 
# building blocks.
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
import model_setup_library

#_____________________________________________________________________________________________________________________________________
#
# General functions 
#
#_____________________________________________________________________________________________________________________________________

def saving_best_model(val_loss, min_val_loss, session, directory, saver, not_save_counter):

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        path = saver.save(session, directory)
        save = "Saved"
        not_save_counter = 0
    else: 
        save = "  -  "
        path = "Did not save model"
        not_save_counter += 1

    return save, path, min_val_loss, not_save_counter

def saving_model(session, directory, saver):

    path = saver.save(session, directory)

    return path

def feeding_dict(placeholders, values, train_flag, train_flag_value):
    placeholders.append(train_flag)
    # values.append(train_flag_value)
    if len(placeholders) == 1: return{placeholders[0]: values}
    else: return {key: value for key, value in zip(placeholders, values + (train_flag_value,))}


#_____________________________________________________________________________________________________________________________________
#
# Layer functions
#
#_____________________________________________________________________________________________________________________________________

def fully_connected_layer(input_x, train_flag, output_dim, activation, dropout_layer, lib, name, print_layer=True):
    reverse_activation_lib = model_setup_library.reverse_activation_lib()

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

        if activation is not None: 
            next_layer = activation(next_layer)
            act = "/" + reverse_activation_lib[activation]
        else: act = ""
        
        if dropout_layer:
            input_x = dropout(input_x, train_flag, keep_prob=lib["keep_prob"], noise_shape=None)
            if lib["keep_prob"] != 1.0: dropout_tag = "/dropout(" + str(lib["keep_prob"]) + ")"
            else: dropout_tag = ""
        else: dropout_tag = ""
        if print_layer: print(f'\t{"Layer type: ":<10}{"Fully connected layer":<20}\t{"Layer name: ":<10}{name+act+dropout_tag:<50}\t{"Layer size: ":<10}{next_layer.shape}')

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


def rnn_layer(input_x, train_flag, input_lengths, output_dim, name, rnn_type="lstm", keep_prob=1.0, scope=None, print_layer=True, **kwargs):

    testing_keep_prob = lambda: 1.0
    training_keep_prob = lambda: keep_prob
    cond_keep_prob = tf.cond(tf.equal(train_flag, tf.constant(True)),
        true_fn=training_keep_prob,
        false_fn=testing_keep_prob)

    with tf.variable_scope(name):

        cell = rnn_cell(output_dim, rnn_type, **kwargs)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=cond_keep_prob, output_keep_prob=keep_prob)
        output, final_state = tf.nn.dynamic_rnn(cell, input_x, sequence_length=input_lengths, dtype=tf.float32, scope=scope)
        if keep_prob != 1.0: dropout_tag = "/dropout(" + str(keep_prob) + ")"
        else: dropout_tag = ""
        if print_layer: 
            print(f'\t{"Layer type: ":<10}{"RNN layer":<20}\t{"Layer name: ":<10}{name+dropout_tag:<50}\t{"Layer size: ":<10}{output.shape}')

    return output, final_state

def multiple_rnn_layers(input_x, train_flag, input_lengths, hidden_layers, lib, layer_offset, print_layer=True, **kwargs):

    for layer_num, layer_size in enumerate(hidden_layers):
        output, final_state = rnn_layer(input_x, train_flag, input_lengths, layer_size, "rnn_layer_{}".format(layer_num+layer_offset), rnn_type=lib["rnn_type"], keep_prob=lib["keep_prob"], print_layer=print_layer, **kwargs)
        input_x = output

    return output, final_state

def conv_layer(input_x, train_flag, kernel, num_input_channels, filt, strides, pool_func, pool_size, activation, dropout_layer, name, lib, print_layer, **kwargs):
    reverse_activation_lib = model_setup_library.reverse_activation_lib()

    with tf.variable_scope(name):
        (length, width) = kernel
        kernel_shape = (length, width, num_input_channels, filt)
        kernel_tensor = tf.get_variable(
            "kernel_tensor", kernel_shape, 
            tf.float32, initializer=tf.glorot_uniform_initializer() 
            )
        bias_tensor = tf.get_variable(
            "bias_tensor", (filt), tf.float32, 
            initializer=tf.zeros_initializer() 
            )

        next_layer = tf.nn.conv2d(input_x, kernel_tensor, strides=[1, strides[0], strides[1], 1], padding='VALID', **kwargs)
        next_layer = tf.nn.bias_add(next_layer, bias_tensor)

        if activation is not None: 
            next_layer = activation(next_layer)
            act = "/" + reverse_activation_lib[activation]
        else: act = ""
        output_shape = [tf.shape(next_layer)[0], next_layer.get_shape()[-3], next_layer.get_shape()[-2], next_layer.get_shape()[-1]]
        if pool_size is not None: 
            next_layer = pooling_layer(next_layer, pool_func, pool_size)
            pool = "/" + lib["pool_func"] + "_pool"
        else: pool = ""

        if dropout_layer:
            input_x = dropout(input_x, train_flag, keep_prob=lib["keep_prob"], noise_shape=[tf.shape(input_x)[0], 1, 1, tf.shape(input_x)[3]])
            if lib["keep_prob"] != 1.0: dropout_tag = "/dropout(" + str(lib["keep_prob"]) + ")"
            else: dropout_tag = ""
        else: dropout_tag = ""

        if print_layer: print(f'\t{"Layer type: ":<10}{"CNN layer":<20}\t{"Layer name: ":<10}{name+act+pool+dropout_tag:<50}\t{"Layer size: ":<10}{next_layer.shape}')

    return next_layer, output_shape

def pooling_layer(input_x, pool_func, pool_size, **kwargs):

    next_layer = pool_func(input_x, ksize=[1, pool_size[0], pool_size[1], 1], strides=[1, pool_size[0], pool_size[1], 1], padding='VALID', **kwargs)

    return next_layer


def deconv_layer(input_x, train_flag, kernel, num_input_channels, filt, strides, deconv_output_shape, pool_output_shape, activation, dropout_layer, name, lib, print_layer, **kwargs):
    reverse_activation_lib = model_setup_library.reverse_activation_lib()

    with tf.variable_scope(name):

        (length, width) = kernel
        kernel_shape = (length, width, filt, num_input_channels)
        kernel_tensor = tf.get_variable(
            "kernel_tensor", kernel_shape, 
            tf.float32, initializer=tf.glorot_uniform_initializer() 
            )

        bias_tensor = tf.get_variable(
            "bias_tensor", [filt], tf.float32, 
            initializer=tf.zeros_initializer() 
            )

        next_layer = tf.nn.conv2d_transpose(input_x, kernel_tensor, deconv_output_shape, strides=[1, strides[0], strides[1], 1], padding='VALID', **kwargs)
        next_layer = tf.nn.bias_add(next_layer, bias_tensor)

        if activation is not None: 
            next_layer = activation(next_layer)
            act = "/" + reverse_activation_lib[activation]
        else: act = ""
        if pool_output_shape is not None: 
            next_layer = unpooling_layer(next_layer, pool_output_shape)
            pool = "/" + lib["pool_func"] + "_unpool"
        else: pool = ""

        if dropout_layer:
            input_x = dropout(input_x, train_flag, keep_prob=lib["keep_prob"], noise_shape=[tf.shape(input_x)[0], 1, 1, tf.shape(input_x)[3]])
            if lib["keep_prob"] != 1.0: dropout_tag = "/dropout(" + str(lib["keep_prob"]) + ")"
            else: dropout_tag = ""
        else: dropout_tag = ""

        if print_layer: print(f'\t{"Layer type: ":<10}{"CNN layer":<20}\t{"Layer name: ":<10}{name+act+pool+dropout_tag:<50}\t{"Layer size: ":<10}{next_layer.shape}')

        
    return next_layer

def unpooling_layer(input_x, output_shape, **kwargs):

    from tensorflow.keras.backend import repeat_elements

    next_layer = tf.image.resize_image_with_crop_or_pad(
        input_x, target_height=int(output_shape[-3]), 
        target_width=int(output_shape[-2])
        )

    return next_layer

def dropout(x, train_flag, keep_prob=1.0, noise_shape=None):
      
    testing_keep_prob = lambda: 1.0
    training_keep_prob = lambda: keep_prob

    cond_keep_prob = tf.cond(tf.equal(train_flag, tf.constant(True)), true_fn=training_keep_prob, false_fn=testing_keep_prob)

    if noise_shape is None: return tf.nn.dropout(x, cond_keep_prob)
    else: return tf.nn.dropout(x, cond_keep_prob, noise_shape)
#_____________________________________________________________________________________________________________________________________
#
# Model architecture
#
#_____________________________________________________________________________________________________________________________________

def cnn_architecture(input_x, train_flag, enc_layers, enc_strides, pooling_lib, pool_layers, latent_layer, 
    dec_layers, dec_strides, lib, activation_lib, print_layer=True, **kwargs):

    pool_func = pooling_lib[lib["pool_func"]]
    activation = activation_lib[lib["activation"]]
    keep_prob = lib["keep_prob"]

    output_shapes = []
    output_shapes.append([tf.shape(input_x)[0], input_x.get_shape()[-3], input_x.get_shape()[-2], input_x.get_shape()[-1]])

    for layer_num, layer_info in enumerate(enc_layers):
        input_x, output_shape = conv_layer(
            input_x, train_flag, (layer_info[-3], layer_info[-2]), input_x.shape[-1], layer_info[-1], enc_strides[layer_num], pool_func, 
            pool_layers[layer_num], activation if layer_num < len(enc_layers) - 1 else None, True, "conv_layer_{}".format(layer_num+1), lib, print_layer
            )
        output_shapes.append(output_shape)
        if pool_layers[layer_num] is not None: output_shapes.append([tf.shape(input_x)[0], input_x.get_shape()[-3], input_x.get_shape()[-2], input_x.get_shape()[-1]])
        
    input_x = tf.layers.flatten(input_x)
    output_shapes.append([tf.shape(input_x)[0], input_x.get_shape()[-1]])

    if lib["latent_func"] != "default":
        if lib["latent_func"] == "fc": 
            func = fc_architecture
            latent_model = func(input_x, train_flag, lib["latent_enc"], latent_layer, lib["latent_dec"], lib, activation_lib, 0, print_layer, True)
            input_x = latent_model["output"]
            latent = latent_model["latent"]
            if output_shapes[-1][-1] not in lib["latent_dec"]: 
                input_x = fully_connected_layer(input_x, train_flag, output_shapes.pop()[-1], None, False, lib, "fc_layer_{}".format(len(lib["latent_enc"])+len(lib["latent_dec"])+1), print_layer)

    else: 
        input_x = fully_connected_layer(input_x, train_flag, latent_layer, None, False, lib, "fc_layer_{}".format(0), print_layer)
        latent = input_x
        input_x = fully_connected_layer(input_x, train_flag, output_shapes.pop()[-1], None, False, lib, "fc_layer_{}".format(1), print_layer)
        

    if input_x.shape != output_shapes[-2]: 
        input_x = tf.reshape(input_x, output_shapes.pop())

    for layer_num, layer_info in enumerate(dec_layers):
        input_x = deconv_layer(
            input_x, train_flag, (layer_info[-3], layer_info[-2]), input_x.shape[-1], layer_info[-1], dec_strides[layer_num], 
            output_shapes.pop(), output_shapes.pop() if len(output_shapes) != 0 else None, activation if layer_num < len(dec_layers) - 1 else tf.nn.sigmoid, 
            True if layer_num < len(dec_layers) - 1 else False, "deconv_layer_{}".format(layer_num+1), lib, print_layer)
       
    output = input_x

    return {"latent": latent, "output": output}

def fc_architecture(input_x, train_flag, enc_layers, latent_layer, dec_layers, lib, activation_lib, layer_offset=0, print_layer=True, sub_network=False):
    activation = activation_lib[lib["activation"]]
    keep_prob = lib["keep_prob"]

    if sub_network is False: 
        input_x = tf.reshape(input_x, [-1, lib["input_dim"]])

    layer_counter = layer_offset

    for layer_num, layer_size in enumerate(enc_layers):
        input_x = fully_connected_layer(input_x, train_flag, layer_size, activation if layer_num < len(enc_layers) - 1 else None, 
            True, lib, "fc_layer_{}".format(layer_counter), print_layer)
        layer_counter += 1

    
    input_x = fully_connected_layer(input_x, train_flag, latent_layer, None, False, lib, "fc_layer_{}".format(layer_counter), print_layer)
    latent = input_x
    layer_counter += 1

    for layer_num, layer_size in enumerate(dec_layers):
        if layer_num < len(dec_layers) - 1: this_activation = activation
        elif sub_network is False: this_activation = tf.nn.softmax
        else: this_activation = None
        input_x = fully_connected_layer(input_x, train_flag, layer_size, this_activation, 
            True if layer_num < len(enc_layers) - 1 else False, lib, "fc_layer_{}".format(layer_counter), print_layer)
        
        layer_counter += 1

    output = input_x

    return {"latent": latent, "output": output}


def rnn_architecture(input_placeholders, train_flag, activation_lib, lib, output_lengths=None, print_layer=True, **kwargs):

    input_x = input_placeholders[0]
    input_lengths = input_placeholders[1]
    max_input_length = (
        tf.reduce_max(input_lengths) if output_lengths is None else
        tf.reduce_max([tf.reduce_max(input_lengths), tf.reduce_max(output_lengths)])
        )
    input_dim = input_x.get_shape().as_list()[-1]
    layer_count = 0
    enc_output, enc_final_state = multiple_rnn_layers(input_x, train_flag, input_lengths, lib["enc"], lib, 0, print_layer, **kwargs)
    layer_count += len(lib["enc"])
    activation = activation_lib[lib["activation"]]

    if lib["rnn_type"] == "lstm":
        input_x, h = enc_final_state
    else:
        input_x = enc_final_state

    if lib["latent_func"] != "default":
        if lib["latent_func"] == "fc": 
            if lib["dec"][0] not in lib["latent_dec"]: lib["latent_dec"].append(lib["dec"][0])
            func = fc_architecture
            latent_model = func(c, train_flag, lib["latent_enc"], lib["latent"], lib["latent_dec"], lib, activation_lib, 0, print_layer, True)
            input_x = latent_model["output"]
            latent = latent_model["latent"]
            fc_layer_count = len(lib["latent_enc"]) + 1 + len(lib["latent_dec"])

    else:
        input_x = fully_connected_layer(input_x, train_flag, lib["latent"], None, False, lib, "fc_layer_{}".format(0), print_layer)
        latent = input_x

        input_x = fully_connected_layer(input_x, train_flag, lib["dec"][0], activation, False, lib, "fc_layer_{}".format(1), print_layer)
        # input_x = activation(input_x)
        fc_layer_count = 2
        
    
    layer_to_dec_dim = input_x.get_shape().as_list()[-1]
    dec_input = tf.reshape(tf.tile(input_x, [1, max_input_length]), [-1, max_input_length, layer_to_dec_dim])

    dec_output, dec_final_state = multiple_rnn_layers(
        dec_input, train_flag, input_lengths if output_lengths is None else output_lengths, lib["dec"], 
        lib, layer_count, print_layer, **kwargs)
    layer_count += len(lib["dec"])

    dec_output = tf.reshape(dec_output, [-1, lib["dec"][-1]])
    dec_output = fully_connected_layer(dec_output, train_flag, input_dim, None, False, lib, "fc_layer_{}".format(fc_layer_count), print_layer)
    dec_output = tf.reshape(dec_output, [-1, max_input_length, input_dim])
    
    return {"latent": latent, "output": dec_output}

def cnn_classifier_architecture(input_x, train_flag, enc_layers, enc_strides, pooling_lib, pool_layers, latent_layer, 
    lib, activation_lib, print_layer=True, **kwargs):

    pool_func = pooling_lib[lib["pool_func"]]
    activation = activation_lib[lib["activation"]]
    keep_prob = lib["keep_prob"]

    for layer_num, layer_info in enumerate(enc_layers):
        input_x, output_shape = conv_layer(
            input_x, train_flag, (layer_info[-3], layer_info[-2]), input_x.shape[-1], layer_info[-1], enc_strides[layer_num], pool_func, 
            pool_layers[layer_num], activation if layer_num < len(enc_layers) - 1 else None, True, "conv_layer_{}".format(layer_num+1), lib, print_layer
            )
    
    input_x = tf.layers.flatten(input_x)

    if lib["latent_func"] != "default":
        if lib["latent_func"] == "fc": 
            func = fc_architecture
            latent_model = func(input_x, train_flag, lib["latent_enc"], latent_layer, [], lib, activation_lib, 0, print_layer, True)
            input_x = latent_model["output"]
            latent = input_x
            layer_counter = len(lib["latent_enc"]) + len(lib["latent_dec"])

    else: 
        input_x = fully_connected_layer(input_x, train_flag, latent_layer, None, False, lib, "fc_layer_{}".format(0), print_layer)
        latent = input_x
        
        layer_counter = 1
    
    input_x = fully_connected_layer(input_x, train_flag, lib["num_classes"], None, False, lib, "fc_layer_{}".format(layer_counter), print_layer)
    output = input_x


    return {"latent": latent, "output": output}

def fc_classifier_architecture(input_x, train_flag, enc_layers, latent_layer, lib, activation_lib, print_layer=True):
    activation = activation_lib[lib["activation"]]
    keep_prob = lib["keep_prob"]

    input_x = tf.reshape(input_x, [-1, lib["input_dim"]])

    for layer_num, layer_size in enumerate(enc_layers):
        input_x = fully_connected_layer(input_x, train_flag, layer_size, activation if layer_num < len(enc_layers) - 1 else None, 
            True, lib, "fc_layer_{}".format(layer_num), print_layer)
        
    input_x = fully_connected_layer(input_x, train_flag, latent_layer, None, False, lib, "fc_layer_{}".format(len(enc_layers)), print_layer)
    latent = input_x

    input_x = fully_connected_layer(input_x, train_flag, lib["num_classes"], None, False, lib, "fc_layer_{}".format(len(enc_layers)+1), print_layer)
    output = input_x

    return {"latent": latent, "output": output}


def rnn_classifier_architecture(input_placeholders, train_flag, activation_lib, lib, print_layer=True, **kwargs):

    input_x = input_placeholders[0]
    input_lengths = input_placeholders[1]
    
    enc_output, enc_final_state = multiple_rnn_layers(input_x, train_flag, input_lengths, lib["enc"], lib, 0, print_layer, **kwargs)

    if lib["rnn_type"] == "lstm":
        input_x, h = enc_final_state
    else:
        input_x = enc_final_state

    if lib["latent_func"] != "default":
        if lib["latent_func"] == "fc": 
            if lib["dec"][0] not in lib["latent_dec"]: lib["latent_dec"].append(lib["dec"][0])
            func = fc_architecture
            latent_model = func(input_x, train_flag, lib["latent_enc"], lib["latent"], [], lib, activation_lib, 0, print_layer, True)
            input_x = latent_model["output"]
            latent = latent_model["latent"]
            layer_counter = len(lib["latent_enc"]) + len(lib["latent_dec"])

    else: 
        input_x = fully_connected_layer(input_x, train_flag, lib["latent"], None, False, lib, "fc_layer_{}".format(0), print_layer)
        latent = input_x
        latent_counter = 1

    input_x = fully_connected_layer(input_x, train_flag, lib["num_classes"], None, False, lib, "fc_layer_{}".format(latent_counter), print_layer)
    output = input_x
    
    return {"latent": latent, "output": output}

def siamese_cnn_architecture(input_x, train_flag, enc_layers, enc_strides, pooling_lib, pool_layers, latent_layer, 
    lib, activation_lib, print_layer=True, **kwargs):

    pool_func = pooling_lib[lib["pool_func"]]
    activation = activation_lib[lib["activation"]]
    keep_prob = lib["keep_prob"]

    for layer_num, layer_info in enumerate(enc_layers):
        input_x, output_shape = conv_layer(
            input_x, train_flag, (layer_info[-3], layer_info[-2]), input_x.shape[-1], layer_info[-1], enc_strides[layer_num], pool_func, 
            pool_layers[layer_num], activation if layer_num < len(enc_layers) - 1 else None, True, "conv_layer_{}".format(layer_num+1), lib, print_layer
            )

    input_x = tf.layers.flatten(input_x)

    if lib["latent_func"] != "default":
        if lib["latent_func"] == "fc": 
            func = fc_architecture
            latent_model = func(input_x, train_flag, lib["latent_enc"], latent_layer, [], lib, activation_lib, 0, print_layer, True)
            input_x = latent_model["output"]
            latent = input_x

    else: 
        input_x = fully_connected_layer(input_x, train_flag, latent_layer, None, False, lib, "fc_layer_{}".format(0), print_layer)
        latent = input_x
    
    latent = tf.nn.l2_normalize(latent, axis=1)

    return {"output": latent}

def siamese_fc_architecture(input_placeholders, train_flag, lib, activation_lib, print_layer=True):
    activation = activation_lib[lib["activation"]]
    keep_prob = lib["keep_prob"]

    input_x = input_placeholders[0]
    input_x = tf.reshape(input_x, [-1, lib["input_dim"]])

    for layer_num, layer_size in enumerate(lib["enc"]):
        input_x = fully_connected_layer(input_x, train_flag, layer_size, activation if layer_num < len(lib["enc"]) - 1 else None, 
            True, lib, "fc_layer_{}".format(layer_num), print_layer)

    input_x = fully_connected_layer(input_x, train_flag, lib["latent"], None, False, lib, "fc_layer_{}".format(len(lib["enc"])), print_layer)
    latent = tf.nn.l2_normalize(input_x, axis=1)

    return {"output": latent}

def siamese_rnn_architecture(input_placeholders, train_flag, activation_lib, lib, print_layer=True, **kwargs):

    input_x = input_placeholders[0]
    input_lengths = input_placeholders[1]
    
    enc_output, enc_final_state = multiple_rnn_layers(input_x, train_flag, input_lengths, lib["enc"], lib, 0, print_layer, **kwargs)

    if lib["rnn_type"] == "lstm":
        input_x, h = enc_final_state
    else:
        input_x = enc_final_state

    if lib["latent_func"] != "default":
        if lib["latent_func"] == "fc": 
            
            func = fc_architecture
            latent_model = func(input_x, train_flag, lib["latent_enc"], lib["latent"], [], lib, activation_lib, 0, print_layer)
            input_x = latent_model["output"]
            latent = latent_model["latent"]

    else:   
        input_x = fully_connected_layer(input_x, train_flag, lib["latent"], None, False, lib, "fc_layer_{}".format(0), print_layer)
        latent = input_x
        
    latent = tf.nn.l2_normalize(latent, axis=1)

    return {"output": latent}

def model_architecture(input_placeholders, lib, activation_lib, **kwargs):

    if lib["architecture"] == "cnn": 
        input_x = tf.reshape(input_placeholders[0], [-1, 28, 28, 1])
    if lib["architecture"] == "fc": 
        input_x = tf.reshape(input_placeholders[0], [-1, 784])
    if lib["architecture"] == "rnn": 
        input_x = input_placeholders[0]
        input_lengths = input_placeholders[1]
        if lib["get_pairs"]: 
            output_lengths = input_placeholders[2]
            max_input_length = tf.reduce_max([tf.reduce_max(input_lengths), tf.reduce_max(output_lengths)])
        else: max_input_length = tf.reduce_max(input_lengths)
        input_dim = input_x.get_shape().as_list()[-1]

    next_layer = input_x
    num_in_channels = 1
    layer_nums = np.zeros(5)

    for i in range(len(lib["layer_order"])):
        layer = lib["layer_order"][i]
        layer_name = "layer_{}".format(int(np.sum(layer_nums)))

        if layer.split(".")[0] == "conv": 
            if len(next_layer.shape) <= 2: 
                next_layer = tf.reshape(next_layer, [-1, 28, 28, num_in_channels])
            next_layer = cnn_layer(next_layer, num_in_channels, lib[layer_name], "conv_layer_{}".format(layer_nums[0]), **kwargs)
            num_in_channels = next_layer.shape[3]
            layer_nums[0] += 1
        elif layer.split(".")[0] == "pool": 
            next_layer = pooling_layer(next_layer, lib[layer_name], "pool_layer_{}".format(layer_nums[1]), **kwargs)
            layer_nums[1] += 1
        elif layer.split(".")[0] == "act":
            activation = activation_lib[layer.split(".")[1]]
            next_layer = activation(next_layer)
            layer_nums[4] += 1
        elif layer.split(".")[0] == "fc":
            if len(next_layer.shape) > 2:
                num_elements_in_feat = next_layer.get_shape()[1:].num_elements()
                next_layer = tf.reshape(next_layer, [-1, num_elements_in_feat])
            next_layer = fully_connected_layer(next_layer, lib[layer_name][0], "fully_connected_layer_{}".format(layer_nums[2]))
            layer_nums[2] += 1
        elif layer.split(".")[0] == "rnn":

            if len(next_layer.shape) < 3:
                next_layer = tf.reshape(tf.tile(next_layer, [1, max_input_length]), [-1, max_input_length, lib[layer_name]])
            if lib["get_pairs"] and i > lib["latent_layer_num"]-1: this_input_lengths = output_lengths
            else: this_input_lengths = input_lengths

            layer_outputs, layer_final_state = rnn_layer(input_x, this_input_lengths, lib[layer_name], "rnn_layer_{}".format(layer_nums[3]), 
                rnn_type=layer.split(".")[1], keep_prob=lib["keep_prob"], **kwargs)
            layer_nums[3] += 1

            if i == len(lib["layer_order"])-1:
                mask = tf.sign(tf.reduce_max(tf.abs(layer_outputs), 2))
                layer_outputs = tf.reshape(layer_outputs, [-1, lib[layer_name]])
                layer_outputs = fully_connected_layer(layer_outputs, input_dim, "linear_decoder_output_layer")
                layer_outputs = tf.reshape(layer_outputs, [-1, max_input_length, input_dim])
                layer_outputs *= tf.expand_dims(mask, -1)
                next_layer = layer_outputs

            else: 
                if lib["layer_order"][i+1].split(".")[0] == "rnn" and i != lib["latent_layer_num"]-1:
                    next_layer = layer_outputs
                elif lib["layer_order"][i+1].split(".")[0] == "fc" or i == lib["latent_layer_num"]-1:
                    if layer.split(".")[1] == "lstm": next_layer, h = layer_final_state
                    else: next_layer = layer_final_state

        if i == lib["latent_layer_num"]-1: latent = next_layer

    if lib["architecture"] == "rnn":
        return{"output": next_layer, "latent": latent, "mask": mask}
    else:
        return{"output": next_layer, "latent": latent}
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

    print("\tRestoring {}-model values from {}...\n".format(lib["model_name"], model_fn))
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
                
def rnn_model_values(states, input_placeholders, data_it, lib, model_fn, labels=None, keys=None):

    input_embeddings = {}
    output_embeddings = {}
    latent_embeddings = {}

    i = 0

    print("\tRestoring {}-model values from {}...\n".format(lib["model_name"], model_fn))
    saver = tf.train.Saver()
    with tf.Session() as sesh:
        saver.restore(sesh, model_fn)
        for feats, lengths in data_it:
            feed_dict = {X: feats, X_lengths: lengths}

            outputs = sesh.run([states[0]], feed_dict)[0]
            latents = sesh.run([states[1]], feed_dict)[0]

            if lib["data_type"] == "buckeye":

                for i in range(len(latents)):
                    output_embeddings[keys[i]] = outputs[i]
                    latent_embeddings[keys[i]] = latents[i]
                    input_embeddings[keys[i]] = feats[i]
            else:
                    input_embeddings["embeddings"] = np.array(feats)
                    input_embeddings["labels"] = np.array(labels)
                    output_embeddings["embeddings"] = np.array(outputs)
                    output_embeddings["labels"] = input_embeddings["labels"]
                    latent_embeddings["embeddings"] = np.array(latents)
                    latent_embeddings["labels"] = input_embeddings["labels"]

    return input_embeddings, output_embeddings, latent_embeddings
                