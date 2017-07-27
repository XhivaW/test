# -*- encoding:utf-8 -*-
import tensorflow as tf 
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl
import pandas as pd 
import numpy as np 
from ulti import *

class GanModel():
    def __init__(self):
        real_result = discriminater_conv(x, kernel)
        real_loss = tf.nn.softmax_cross_entropy_with_logits(real_y, real_result)/batch_size
    
        fake_image = generator_deconv(noise)
        fake_result = discriminater_conv(fake_image, kernel)
        fake_loss = tf.nn.softmax_cross_entropy_with_logits(fake_y, fake_result)/batch_size
    
        margin = 20
        D_loss = margin - fake_loss + real_loss
        G_loss = fake_loss
    
        self.train_op_G = optimizer(G_loss, 'generator')
        self.train_op_D = optimizer(D_loss, 'discriminater')

        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth=True

        self.sess = tf.Session(config=config)

    def generator_deconv(self, z, kernel):
        '''
        inputs:
                self: inherited from class
                z: random tensor in shape [batch_num,random_dim]
                kernel: convonlution kernel in shape [height, width, inchannel, outchannel]
        '''
        with tf.variable_scope("generator"):
            result = tfcl.fully_connected(z,13*4, activation_fn=tf.nn.relu, normalizer_fn=tfcl.batch_norm)
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


    def discriminater_conv(self, z, kernel, reuse=False):
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


    def optimizer(self, loss, d_or_g):
        optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
        var_list = [v for v in tf.tranable_variables() if v.name.startswith(d_or_g)]
        gradient = optim.compute_gradients(loss, var_list=var_list)
        return optim.apply_gradients(gradient)

    def train(self, real_folder, training_epoches=20000, batch_size=75):
        i = 0
        self.sess.run(tf.global_variables_initializer())

        for epoch in xrange(training_epoches):
            n_d = 100 if epoch<25 or (epoch+1)%500 == 0 else 5
            for _ in xrange(n_d):
