# -*- encoding:utf-8 -*-
import tensorflow as tf 
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl
import pandas as pd 
import numpy as np 
#from test import *

kernel = tf.constant(1.0, shape=[4,4,1,1])
protype = tf.constant(1.0, shape=[3,4])
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth=True

with tf.variable_scope("generator"):
    t1 = tf.get_variable("weights",shape=[4,52])
    t2 = tf.get_variable("biases", shape=[52])

wtf = generator_deconv(protype,kernel)

with tf.Session(config=config) as sess:
    sess.run(init)
    sess.run(tf.shape(wtf))

def generator_deconv(z, kernel):
    '''
    inputs:
            self: inherited from class
            z: random tensor in shape [1,]
    '''
    with tf.variable_scope("generator", reuse=True):
        weights = tf.get_variable("weights")
        biases = tf.get_variable("biases")
        result = tf.matmul(z, weights)
        result = tf.add(result, biases)
        result = tf.reshape(result, tf.stack([tf.shape(result)[0],13,4,1]))
        result = tf.reshape(result, tf.stack([tf.shape(result)[0],13,4,1]))
        result = tf.nn.conv2d_transpose(result, kernel, 
                output_shape=[tf.shape(result)[0],25,8,1], 
                strides=[1,2,2,1], 
                padding="SAME")
        result = tf.nn.conv2d_transpose(result, kernel, 
                output_shape=[tf.shape(result)[0],50,15,1], 
                strides=[1,2,2,1], 
                padding="SAME")
        result = tf.nn.conv2d_transpose(result, kernel, 
                output_shape=[tf.shape(result)[0],100,30,1], 
                strides=[1,2,2,1], 
                padding="SAME")    
        return result
