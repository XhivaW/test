# -*- encoding:utf-8 -*-
import tensorflow as tf 
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl
import pandas as pd 
import numpy as np 
import sys
from datetime import datetime
from PIL import Image
from ulti import *

class GanModel():
    def __init__(self, data, load_model):
        self.load_model = load_model
        self.data = data

        self.output_w = 100
        self.output_h = 30

        self.x = tf.placeholder(tf.float32, shape=[None,30,100,1])
        #self.real_label = tf.placeholder(tf.float32, shape=[None,1])
        #self.fake_label = tf.placeholder(tf.float32, shape=[None,1])
        self.z = tf.placeholder(tf.float32, shape=[None,100])
        with tf.variable_scope("generator"):
            self.kernel_g = tf.get_variable("kernel_g", shape=[4,4,1,1])
        with tf.variable_scope("discriminater"):
            self.kernel_d = tf.get_variable("kernel_d", shape=[4,4,1,1])

        # nets
        self.fake_image = self.generator_deconv(self.z)

        self.d_result = self.discriminater_conv(self.x)
        self.g_result = self.discriminater_conv(self.fake_image, reuse=True)

        # loss
        '''
        self.g_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_label, logits=self.g_result)
        self.d_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_label, logits=self.d_result) \
                    + tf.nn.softmax_cross_entropy_with_logits(labels=self.fake_label, logits=self.g_result)
        '''
        self.g_loss = - tf.reduce_mean(self.g_result)
        self.d_loss = - tf.reduce_mean(self.d_result) + tf.reduce_mean(self.g_result)


        self.g_solver = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=self.get_var_list("generator"))
        self.d_solver = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=self.get_var_list("discriminater"))
        #self.g_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.g_loss, var_list=self.get_var_list("generator"))
        #self.d_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.d_loss, var_list=self.get_var_list("discriminater"))


        # clip
        # Why we need this???
        # Why we limit from -0.01 to 0.01 here??
        # see [https://zhuanlan.zhihu.com/p/25071913]
        self.clip_d = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.get_var_list("discriminater")]

        # GPU config ans session
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth=True

        self.sess = tf.Session(config=config)

        self.saver = tf.train.Saver()


    def get_var_list(self, scope_name):
        return [var for var in tf.global_variables() if scope_name in var.name]

    def lrelu(self, x, leak=0.2):
            return tf.maximum(x, leak*x)
    '''
    def deconv2d(self, z, output_shape, activation_fn=tf.nn.relu, k_h=4, k_w=4, 
                name="deconv2d", strides=[1,2,2,1], 
                padding="SAME", stddev=0.02):
        kernel = tf.get_variable(name+"_kernel", 
                shape=[k_h,k_w,output_shape[-1],1], 
                initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(z, kernel, output_shape=output_shape,
                strides=strides, padding=padding, activation_fn=activation_fn,
                weights_initializer=tf.random_normal_initializer(stddev=stddev))

        #biases = tf.get_variable(name+"_biases", [output_shape[-1]], 
        #                        initializer=tf.constant_initializer(0.0))
        #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.shape(deconv))

        return deconv
    '''

    def deconv2d(self, z, output_size=1, name="deconv2d", kernel_size=4, 
        stride=2, activation_fn=tf.nn.relu, padding="SAME", stddev=0.02, 
        normalizer_fn=None):
        deconv = tfcl.conv2d_transpose(z, output_size, kernel_size, stride=stride, 
                    activation_fn=activation_fn, normalizer_fn=normalizer_fn, 
                    padding=padding, weights_initializer=tf.random_normal_initializer(0,stddev))
        return deconv

    '''
    def conv2d(self, z, output_dim, activation_fn=None, k_h=4, k_w=4, 
              name="conv2d", strides=[1,2,2,1], 
              padding="SAME", stddev=0.02):
        if activation_fn is None:
            activation_fn = self.lrelu

        kernel = tf.get_variable(name+"_kernel", 
                shape=[k_h,k_w,z.get_shape()[-1], output_dim], 
                initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(z, kernel, strides=strides, padding=padding, 
                            activation_fn=activation_fn, 
                            weights_initializer=tf.random_normal_initializer(stddev=stddev))

        #biases = tf.get_variable(name+"_biases", [output_dim], 
        #                        initializer=tf.constant_initializer(0.0))
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv
    '''

    def conv2d(self, z, output_size=1, name="conv2d", kernel_size=4, 
        stride=2, activation_fn=None, padding="SAME", stddev=0.02, 
        normalizer_fn=None):
        if activation_fn is None:
            activation_fn = self.lrelu

        conv = tfcl.conv2d(z, output_size, kernel_size, stride=stride, 
                activation_fn=activation_fn, normalizer_fn=normalizer_fn, 
                padding=padding, weights_initializer=tf.random_normal_initializer(0,stddev))

        return conv

    def generator_deconv(self, z):
        '''
        inputs:
                self: inherited from class
                z: random tensor in shape [batch_num,random_dim]
                kernel: convonlution kernel in shape [height, width, inchannel, outchannel]
        '''
        with tf.variable_scope("generator"):
            '''
            #result = tfcl.fully_connected(z,30, activation_fn=self.lrelu, normalizer_fn=tfcl.batch_norm)
            result = tfcl.fully_connected(z,13*4, activation_fn=self.lrelu, normalizer_fn=tfcl.batch_norm)
            result = tf.reshape(result, tf.stack([tf.shape(result)[0],4,13,1]))
            result = tf.nn.conv2d_transpose(result, self.kernel_g, 
                    output_shape=[tf.shape(result)[0],25,8,1], 
                    strides=[1,2,2,1], 
                    padding="SAME")
            result = tf.nn.conv2d_transpose(result, self.kernel_g, 
                    output_shape=[tf.shape(result)[0],50,15,1], 
                    strides=[1,2,2,1], 
                    padding="SAME")
            result = tf.nn.conv2d_transpose(result, self.kernel_g, 
                    output_shape=[tf.shape(result)[0],100,30,1], 
                    strides=[1,2,2,1], 
                    padding="SAME")
            
            result = tfcl.fully_connected(result,25*8, activation_fn=self.lrelu, normalizer_fn=tfcl.batch_norm)
            result = tfcl.fully_connected(result,50*15, activation_fn=self.lrelu, normalizer_fn=tfcl.batch_norm)
            result = tfcl.fully_connected(result,100*30, activation_fn=tf.tanh, normalizer_fn=tfcl.batch_norm)
            result = tf.reshape(result, tf.stack([tf.shape(result)[0],100,30,1]))
            
            result = tfcl.fully_connected(z, 13*4, activation_fn=self.lrelu, normalizer_fn=tfcl.batch_norm)
            result = tf.reshape(result, tf.stack([tf.shape(result)[0],4,13,1]))
            result = self.deconv2d(result, [tf.shape(result)[0],25,8,1], name="g_deconv_1")
            result = self.deconv2d(result, [tf.shape(result)[0],50,15,1], name="g_deconv_2")
            result = self.deconv2d(result, [tf.shape(result)[0],100,30,1], 
                                    name="g_deconv_3", activation_fn=tf.tanh)
            '''
            result = tfcl.fully_connected(z, 4*13*8, activation_fn=self.lrelu, normalizer_fn=tfcl.batch_norm)
            result = tf.reshape(result, tf.stack([tf.shape(z)[0],4,13,8]))
            result = self.deconv2d(result, output_size=4, normalizer_fn=tfcl.batch_norm)
            result = self.deconv2d(result, output_size=2)
            result = self.deconv2d(result)
            return result


    def discriminater_conv(self, z, reuse=False):
        '''
        inputs:
                self: inherited from class
                z: random tensor in shape [batch_num,height, width, channel]
                kernel: convonlution kernel in shape [height, width, inchannel, outchannel]
        '''
        with tf.variable_scope("discriminater") as vs:
            if reuse:
                vs.reuse_variables()
            '''
            result = tf.nn.conv2d(z, self.kernel_d, 
                    strides=[1,2,2,1],
                    padding="SAME")
            result = tf.nn.conv2d(result, self.kernel_d, 
                    strides=[1,2,2,1],
                    padding="SAME")
            result = tf.nn.conv2d(result, self.kernel_d, 
                    strides=[1,2,2,1],
                    padding="SAME")
            result = tf.reshape(result, tf.stack([tf.shape(result)[0],13*4]))
            #result = tfcl.fully_connected(result, 256, activation_fn=self.lrelu, normalizer_fn=tfcl.batch_norm)
            result = tfcl.fully_connected(result, 128, activation_fn=self.lrelu, normalizer_fn=tfcl.batch_norm)
            result = tfcl.fully_connected(result, 16, activation_fn=self.lrelu, normalizer_fn=tfcl.batch_norm)
            result = tfcl.fully_connected(result, 2, activation_fn=None)
            
            result = self.conv2d(z, 1, name="d_conv_1")
            result = self.conv2d(result, 1, name="d_conv_2")
            result = self.conv2d(result, 1, name="d_conv_3")
            '''
            result = self.conv2d(z,output_size=2)
            result = self.conv2d(result,output_size=4)
            result = self.conv2d(result,output_size=8, normalizer_fn=tfcl.batch_norm)
            result = tf.reshape(result, tf.stack([tf.shape(z)[0],13*4*8]))
            result = tfcl.fully_connected(result, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0,0.02))
            return result

    def train(self):
        i = 0
        step = 0
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            ckpt = tf.train.get_checkpoint_state('./saved_model/')
            step = int(ckpt.model_checkpoint_path.split('-')[-1])
            print("Load previous model from {}.\nStart iteration from {}.".format(ckpt.model_checkpoint_path, step))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        while True:
            if step<25 or (step+1)%500 == 0:
                n_d = 100
            else:
                n_d = 5
            #n_d = 50 if step<25 or (step+1)%500 == 0 else 2
            for __ in xrange(n_d):
                '''
                real_batch, real_label, fake_label = self.data.get_next_batch()
                #print("===============\n{}\t{}\n=============".format(real_label.shape, fake_label.shape))
                if np.random.uniform(0,1)<0.25:
                    real_label, fake_label = fake_label, real_label
                self.sess.run(self.clip_d)
                self.sess.run(self.d_solver,
                    feed_dict={self.x: real_batch, 
                                self.real_label: real_label, 
                                self.fake_label: fake_label, 
                                self.z: generate_z()})
                '''
                real_batch, _1, _2= self.data.get_next_batch()
                self.sess.run(self.clip_d)
                self.sess.run(self.d_solver,
                    feed_dict={self.x: real_batch, 
                                self.z: generate_z()})
            '''
            self.sess.run(
                self.g_solver, 
                feed_dict={self.real_label: real_label, 
                            self.fake_label: fake_label, 
                            self.z: generate_z()})
            '''
            self.sess.run(
                self.g_solver, 
                feed_dict={self.z: generate_z()})

            if step % 50 == 0 or (step < 100):
                d_loss_curr = self.sess.run(tf.reduce_mean(self.d_loss), 
                        feed_dict={self.x: real_batch, 
                                self.z: generate_z()})

                g_loss_curr = self.sess.run(tf.reduce_mean(self.g_loss), 
                        feed_dict={self.z: generate_z()})

                print('Current step: {}, D loss: {}, G loss: {}'.format(step, d_loss_curr, g_loss_curr))

                if (step % 500 == 0) and step > 0:

                    time = datetime.now().strftime('%Y%m%d_%H%M%S')

                    sample = self.sess.run(self.fake_image, 
                                feed_dict={self.z: generate_z(batch_size=1)})
                    #print sample.shape
                    #print '\n',type(sample), sample.shape,'\n'
                    pic = self.data.data2pic(sample)   
                    print("Generate fake sample to ./fake_sample/time_{}_{}_step_{}.jpg".format(time, i, step))
                    pic.save("./fake_sample/time_{}_{}_step_{}.jpg".format(time, i, step))
                    i += 1


                    if (step % 2000 == 0 and step > 0) or (step == 1000):

                        print("Save current model to ./saved_model/gan_{}_{}.model".format(step, time))
                        self.saver.save(self.sess, 
                            './saved_model/gan_step_{}_time_{}.model'.format(step, time), 
                            global_step=step)

            step += 1


if __name__ == '__main__':

    load_model = True if '-l' in sys.argv else False
    data = real_data()

    gan = GanModel(data, load_model)
    gan.train()



'''
    def optimizer(self, loss, d_or_g):
        optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
        var_list = [v for v in tf.tranable_variables() if v.name.startswith(d_or_g)]
        gradient = optim.compute_gradients(loss, var_list=var_list)
        return optim.apply_gradients(gradient)
'''