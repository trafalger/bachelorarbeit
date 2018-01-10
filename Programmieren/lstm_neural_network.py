#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 20:13:46 2018

@author: trafalger
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
from scipy.io.wavfile import read
import librosa
import matplotlib.pyplot as plt
import wave, sys, pyaudio
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta
import textgrid
from tensorflow.python.ops import rnn, rnn_cell

# Read sound
n_examples = 1
srate, sig = read('004.wav.wav')
print('srate (Hz):', srate)
print('duration (sec):', len(sig)/srate)
plt.plot(np.arange(len(sig))/srate, sig)
plt.title('004.wav.wav')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.show()

winlen = 0.025
winstep = 0.01
numcep = 13
mfcc_raw = mfcc(sig, srate, winlen, winstep, numcep,
                appendEnergy = True) # 13-d MFCC
mfcc_deriv1 = delta(mfcc_raw, N = 2) # 1st deriv
mfccs = np.concatenate((mfcc_raw, mfcc_deriv1), axis=1).astype(np.float32)
# mfccs = mfcc_raw
plt.imshow(np.rot90(mfccs, axes=(0,1)), aspect='auto')
plt.title('MFCC values (26 dimension)')
plt.xlabel('Time (msec)')
plt.ylabel('Coefficients')
plt.show()

print('Input dimension:',mfccs.shape)

# Read textgrid
T = textgrid.TextGrid()
T.read('004_wav.TextGrid')
w_tier = T.getFirst('Vokale').intervals
time_mark = (winlen/2) + winstep*np.arange(0, mfccs.shape[0])
time_mark = time_mark.astype('float32')


words_raw = []
for t in time_mark:
    for ival in range(len(w_tier)):
        if t > w_tier[ival].bounds()[0] and t <= w_tier[ival].bounds()[1]:
            words_raw.append(w_tier[ival].mark)

words_list = list(set(words_raw)) # unique word list
words_idx = {w: i for i, w in enumerate(words_list)}
words_data = [words_idx[w] for w in words_raw]
words_data_onehot = tf.one_hot(words_data,
                              depth = len(words_list),
                              on_value = 1.,
                              off_value = 0.,
                              axis = 1,
                              dtype=tf.float32)
with tf.Session() as sess: # convert from Tensor to numpy array
    words_label = words_data_onehot.eval()
print('words_list:',words_list)
print('output dimension:',words_label.shape)
#Hyper Parameters
#learning rate for greedy and descent method
learning_rate = 0.01
#iteration for backprpagation, the higher max_it the higher the progress
max_it = 50

# Network Parameters
n_input_dim = mfccs.shape[1]
n_input_len = mfccs.shape[0]
n_output_dim = words_label.shape[1]
n_output_len = words_label.shape[0]
print 
n_hidden = 300
# TensorFlow graph
# (batch_size) x (time_step) x (input_dimension)
x_data = tf.placeholder(tf.float32, [None,None, n_input_dim])
# (batch_size) x (time_step) x (output_dimension)
y_data = tf.placeholder(tf.float32, [ None,None,None])
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output_dim]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output_dim]))
}

def RNN(x, weights, biases):
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,reuse=True) # Make RNNCell
    outputs, states = tf.nn.dynamic_rnn(cell, x, time_major=False, dtype=tf.float32)


    '''
    **Notes on tf.nn.dynamic_rnn**

    - 'x' can have shape (batch)x(time)x(input_dim), if time_major=False or 
                         (time)x(batch)x(input_dim), if time_major=True
    - 'outputs' can have the same shape as 'x'
                         (batch)x(time)x(input_dim), if time_major=False or 
                         (time)x(batch)x(input_dim), if time_major=True
    - 'states' is the final state, determined by batch and hidden_dim
    '''
    
    # outputs[-1] is outputs for the last example in the mini-batch
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x_data, weights, biases)
cost = tf.reduce_mean(tf.squared_difference(pred[-1], y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
test =[]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= max_it:
        loss = 0
        for i in range(n_examples):
            key = 's' + str(i + 1)
            x_train = mfccs.reshape((1, mfccs.shape[0], n_input_dim))
            y_train = words_label.reshape((1, words_label.shape[0], n_output_dim))
            c, _ = sess.run([cost, optimizer], feed_dict={x_data: x_train, y_data: y_train})
            loss += c
            test.append(RNN(x_data, weights, biases))
        mean_mse = loss / n_examples

        print('Epoch =', str(step), '/', str(max_it),
              'Cost = ', '{:.5f}'.format(mean_mse))
        step += 1
        pred_out = sess.run(pred, feed_dict={x_data: x_train})
        pred_out = np.argmax(pred_out, 1)

        plt.subplot(211)
        plt.plot(words_data)
        plt.yticks([0, 1, 2, 3], words_list)
        plt.subplot(212)
        plt.plot(pred_out)
        plt.yticks([0, 1, 2, 3], words_list)
        plt.show()
        
# Test
with tf.Session() as sess:
    sess.run(init)
    pred_out = sess.run(pred, feed_dict={x_data: x_train})
    pred_out = np.argmax(pred_out, 1)
    
    plt.subplot(211)
    plt.plot(words_data)
    plt.yticks([0, 1, 2, 3], words_list)
    plt.subplot(212)
    plt.plot(pred_out)
    plt.yticks([0, 1, 2, 3], words_list)
    plt.show()