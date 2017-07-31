# -*- encoding:utf-8 -*-
import tensorflow as tf 
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl
import pandas as pd 
import numpy as np 
from datetime import datetime

# remove scientific notation
#np.set_printoptions(suppress=True)

#train_prob = 0.95


#train_idx = int(train_df.shape[0]*train_prob)

#x_train = train_df.drop(['target'], axis=1).loc[:train_idx]
#y_train = train_df.target.loc[:train_idx]
#x_test = train_df.drop(['target'], axis=1).loc[train_idx:]
#y_test = train_df.target.loc[train_idx:]
#y_last = train_df['last'].loc[train_idx:]
#y_log10 = train_df.log10.loc[train_idx:]
# shape        
#print('Shape train label: {}\nShape test label: {}\nShape test last: {}'.format(y_train.shape, y_test.shape, y_last.shape))

#y_mean = np.mean(y_train)

#split = int(x_train.shape[0]*0.95)
#x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

class network():
    def __init__(self):
        self.name = 'nn'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            g = tfcl.fully_connected(z, 150, activation_fn=tf.nn.relu, normalizer_fn=tfcl.batch_norm)
            #g = tfcl.fully_connected(g, 300, activation_fn=tf.nn.relu, normalizer_fn=tfcl.batch_norm)
            #g = tfcl.fully_connected(g, 600, activation_fn=tf.nn.relu, normalizer_fn=tfcl.batch_norm)
            g = tfcl.fully_connected(g, 200, activation_fn=tf.nn.tanh)
            g = tfcl.fully_connected(g, 50, activation_fn=tf.nn.tanh, normalizer_fn=tfcl.batch_norm)
            g = tfcl.fully_connected(g, 10, activation_fn=tf.nn.tanh, normalizer_fn=tfcl.batch_norm)
            g = tfcl.fully_connected(g, 5, activation_fn=tf.nn.tanh, normalizer_fn=tfcl.batch_norm)
            g = tfcl.fully_connected(g, 1, activation_fn=None)
            return g

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class tfnn():
    def __init__(self, network, data):
        self.network = network
        self.data = data
        self.test_prob = 0.05

        self.x = tf.placeholder(tf.float32,shape=[None,29])
        self.y = tf.placeholder(tf.float32,shape=[None])
        # nets
        self._y = self.network(self.x)

        # loss
        #self.loss, _ = tf.metrics.mean_squared_error(self.y, self._y)
        #self.loss = tf.reduce_mean(tf.reduce_mean(tf.square(self.y - self._y), axis=1), axis=0)
        self.loss = tf.nn.l2_loss(self.y - self._y)

        self.solver = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss, var_list=self.network.vars)
        #self.solver = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.network.vars)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allocator_type = 'BFC'
        self.config.gpu_options.allow_growth=True

        self.sess = tf.Session(config=self.config)
        self.saver = tf.train.Saver()
        self.result_path = '../result/nn_'

    def train(self, epoches=30, batch_size=266):
        self.sess.run(tf.global_variables_initializer())

        for epoch in xrange(epoches):
            train_df = self.data.sample(frac=1).reset_index(drop=True)
            test_idx = int(train_df.shape[0]*self.test_prob)
            test_idx = 1000 if test_idx>1000 else test_idx
            x_train = np.asarray(train_df.drop(['target'], axis=1).loc[test_idx:])
            y_train = np.asarray(train_df.target.loc[test_idx:])
            x_test = np.asarray(train_df.drop(['target'], axis=1).loc[:test_idx])
            y_test = np.asarray(train_df.target.loc[:test_idx])
            batch_num = x_train.shape[0] // batch_size
            for i in xrange(batch_num):
                self.sess.run(self.solver,
                                feed_dict={
                                    self.x : x_train[i*batch_size:(i+1)*batch_size-10,],
                                    self.y : y_train[i*batch_size:(i+1)*batch_size-10,]})
                if i % 100 ==0:
                    train_loss = self.sess.run(self.loss, 
                                            feed_dict={
                                                self.x : x_train[i*batch_size:(i+1)*batch_size-10,],
                                                self.y : y_train[i*batch_size:(i+1)*batch_size-10,]})
                    valid_loss = self.sess.run(self.loss, 
                                            feed_dict={
                                                self.x : x_train[i*batch_size+64:(i+1)*batch_size,],
                                                self.y : y_train[i*batch_size+64:(i+1)*batch_size,]})
                    print('Epoch: {}, Iter: {},\n\ttrain loss: {}, valid loss: {}'.format(epoch, i, train_loss, valid_loss))
            
            global_num = int(epoch*batch_num*batch_size+i*batch_size)
            
            self.saver.save(self.sess, '../model/test_{}.model'.format(global_num), global_step=global_num)

            test_loss = self.sess.run(self.loss,
                                    feed_dict={
                                    self.x : x_test, self.y : y_test})

            print('Test loss: {}'.format(test_loss))

            prediction = self.sess.run(self._y,
                                    feed_dict={
                                    self.x : x_test})

            if (epoch+1) % 5 == 0:
                goods_id = x_test[:,0].astype(np.int64)
                channel_id = x_test[:,1].astype(np.int64)
                name = self.result_path+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(int(test_loss*10000))+'.result'
                print('Save result as: '+name)
                with open(name,'aw') as f:
                    for j in xrange(len(y_test)):
                        f.write(str(goods_id[j])+','+str(channel_id[j])+','+str(prediction[j][0])+','+str(y_test[j])+'\n')


if __name__=='__main__':
    print("Read train data")
    train_df = pd.read_csv("../data/num_all.pd",sep='\t').sample(frac=1).reset_index(drop=True)

    print('Shape data: {}'.format(train_df.shape))

    net = network()

    model = tfnn(net, train_df)
    model.train()
