# -*- encoding:utf-8 -*-
import tensorflow as tf 
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl
import pandas as pd 
import numpy as np 
from datetime import datetime
from PIL import Image
from ulti import *

class GanModel():
    def __init__(self, data):
        self.data = data
        self.x = tf.placeholder(tf.float32, shape=[None,100,30,1])
        self.real_label = tf.placeholder(tf.float32, shape=[None,2])
        self.z = tf.placeholder(tf.float32, shape=[None,20])
        with tf.variable_scope("generator"):
            self.kernel_g = tf.get_variable("kernel_g", shape=[4,4,1,1])
        with tf.variable_scope("discriminater"):
            self.kernel_d = tf.get_variable("kernel_d", shape=[4,4,1,1])

        # nets
        self.fake_image = generator_deconv(self.z, self.kernel_g)
        self.g_result = discriminater_conv(self.fake_image, self.kernel_d, reuse=True)

        self.d_result = discriminater_conv(self.x, self.kernel_d)


        # loss
        self.g_loss = tf.softmax_cross_entropy_with_logits(self.g_result, self.real_label)
        self.d_loss = tf.softmax_cross_entropy_with_logits(self.d_result, self.real_label) - self.g_loss

        self.g_solver = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=get_var_list("generator"))
        self.d_solver = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=get_var_list("discriminater"))


        # clip
        # Why we need this???
        # Why we limit from -0.01 to 0.01 here??
        self.clip_d = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in get_var_list("discriminater")]

        # GPU config ans session
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth=True

        self.sess = tf.Session(config=config)

        self.saver = tf.train.Saver()


    def get_var_list(self, scope_name):
        return [var for var in tf.global_variables() if scope_name in var.name]


    def generator_deconv(self, z, kernel):
        '''
        inputs:
                self: inherited from class
                z: random tensor in shape [batch_num,random_dim]
                kernel: convonlution kernel in shape [height, width, inchannel, outchannel]
        '''
        with tf.variable_scope("generator"):
            result = tfcl.fully_connected(z,30, activation_fn=tf.nn.relu, normalizer_fn=tfcl.batch_norm)
            result = tfcl.fully_connected(result,13*4, activation_fn=tf.nn.relu, normalizer_fn=tfcl.batch_norm)
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
            #result = tfcl.fully_connected(result, 256, activation_fn=tf.tanh, normalizer_fn=tfcl.batch_norm)
            result = tfcl.fully_connected(result, 128, activation_fn=tf.nn.tanh)
            result = tfcl.fully_connected(result, 16, activation_fn=tf.nn.tanh)
            result = tfcl.fully_connected(result, 2, activation_fn=None)
            return result

'''
    def optimizer(self, loss, d_or_g):
        optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
        var_list = [v for v in tf.tranable_variables() if v.name.startswith(d_or_g)]
        gradient = optim.compute_gradients(loss, var_list=var_list)
        return optim.apply_gradients(gradient)
'''

    def train(self):
        i = 0
        epoch = 0
        self.sess.run(tf.global_variables_initializer())

        while True:
            n_d = 100 if epoch<25 or (epoch+1)%500 == 0 else 5
            for _ in xrange(n_d):
                real_batch, real_label = self.data.get_next_batch()
                self.sess.run(self.clip_d)
                self.sess.run(self.d_solver,
                    feed_dict={self.x: real_batch, 
                                self.real_label: real_label, 
                                self.z: generate_z()})

            self.sess.run(
                self.g_solver, 
                feed_dict={self.z: generate_z()})

            if epoch % 100 == 0 or epoch < 100:
                d_loss_curr = self.sess.run(self.d_loss, 
                        feed_dict={self.x: real_batch, 
                                    self.real_label: real_label,
                                    self.z: generate_z()})

                g_loss_curr = self.sess.run(self.g_loss, 
                        feed_dict={self.z: generate_z()})

                print('Current epoch: {}\n\tD loss: {:.4}, G loss: {:.4}'.format(epoch, d_loss_curr, g_loss_curr))

                if epoch % 500 == 0:
                    sample = self.sess.run(self.fake_image, 
                                feed_dict={self.z: generate_z(batch_size=1)})

                    pic = self.data.data2pic(sample)
                    pic.save("./fake_sample/{}_epoch_{}.jpg".format(str(i),epoch))
                    i += 1


                    if epoch % 1000 == 0:

                        time = datetime.now().strftime('%Y%m%d_%H%M%S')

                        self.saver.save(self.sess, 
                            './saved_model/gan_{}_{}.model'.format(epoch, time), 
                            global_step=epoch)

            epoch += 1


if __name__ == '__main__':

    data = real_data()

    gan = GanModel(data)
    gan.train()