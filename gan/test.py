# -*- encoding:utf-8 -*-
import tensorflow as tf 
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl
import pandas as pd 
import numpy as np 
#from test import *

def discriminater_conv(z, kernel, reuse=False):
    '''
    inputs:
            self: inherited from class
            z: random tensor in shape [batch_num,height, width, channel]
            kernel: convonlution kernel in shape [height, width, inchannel, outchannel]
    '''
    with tf.variable_scope("discriminater") as vs:
        if reuse:
            vs.reuse_variables()
        result = tf.nn.conv2d(z, kernel, 
                strides=[1,2,2,1],
                padding="SAME")
        result = tf.nn.conv2d(result, kernel, 
                strides=[1,2,2,1],
                padding="SAME")
        result = tf.nn.conv2d(result, kernel, 
                strides=[1,2,2,1],
                padding="SAME")
        result = tf.reshape(result, tf.stack([tf.shape(result)[0],13*4]))
        result = tfcl.fully_connected(result, 256, activation_fn=tf.tanh, normalizer_fn=tfcl.batch_norm)
        result = tfcl.fully_connected(result, 128, activation_fn=tf.tanh)
        result = tfcl.fully_connected(result, 16, activation_fn=tf.tanh)
        result = tfcl.fully_connected(result, 2, activation_fn=tf.nn.softmax)
        return result

kernel = tf.constant(1.0, shape=[4,4,1,1])
protype = tf.Variable(tf.random_normal([3,100,30,1]))

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth=True

with tf.variable_scope("generator"):
    t1 = tf.get_variable("weights", shape=[4,52])
    t2 = tf.get_variable("biases", shape=[52])

wtf = discriminater_conv(protype,kernel)
#init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.shape(t1))
    sess.run(tf.shape(t2))
    sess.run(tf.shape(wtf))
