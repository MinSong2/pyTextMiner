#!/bin/env python
#-*- coding: utf8 -*-

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def RNN(_X, _istate, _weights, _biases, n_hidden, n_steps, n_input, early_stop):
    # input _X shape: Tensor("Placeholder:0", shape=(?, n_steps, n_input), dtype=float32)
    # switch n_steps and batch_size, Tensor("transpose:0", shape=(n_steps, ?, n_input), dtype=float32)
    _X = tf.transpose(_X, [1, 0, 2])
    # Reshape to prepare input to hidden activation
    # (n_steps*batch_size, n_input) => (?, n_input), Tensor("Reshape:0", shape=(?, n_input), dtype=float32)
    _X = tf.reshape(_X, [-1, n_input])
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden'] # (?, n_hidden)+scalar(n_hidden,)=(?,n_hidden)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=False)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    # n_steps splits each of which contains (?, n_hidden)
    # ex) [<tf.Tensor 'split:0' shape=(?, n_hidden) dtype=float32>, ... , <tf.Tensor 'split:n_steps-1' shape=(?, n_hidden) dtype=float32>]
    _X = tf.split(_X, n_steps, 0)
    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(cell=lstm_cell, inputs=_X, initial_state=_istate, sequence_length=early_stop)
    final_outputs = []
    for output in outputs :
        # Linear activation
        final_output = tf.matmul(output, _weights['out']) + _biases['out'] # (?, n_classes)
        final_outputs.append(final_output)
    # [<tf.Tensor 'add_1:0' shape=(?, n_classes), ..., <tf.Tensor 'add_n_steps:0' shape=(?, n_classes) dtype=float32>]
    return final_outputs