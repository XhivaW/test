# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd 
import numpy as np 
import os
import random
from PIL import Image
import urllib
import cStringIO

class mnist_data():
    def __init__(self, mnist_folder='./mnist/', batch_size=64):
        self.path = mnist_folder
        self.counter = 0
        self.batch_size = batch_size
        self.data, self.label = self.get_mnist_input()
        self.limit = self.data.shape[0]

    def get_mnist_input(self):
        x = np.load(self.path+'all_x.npy').astype(np.float32)
        y = np.load(self.path+'all_y.npy').astype(np.float32)
        return x, y

    def get_next_batch(self):
        next_batch = []
        while (self.counter+self.batch_size)>=self.limit:
            self.counter = np.random.randint(self.limit)
        tmp = self.data[self.counter:self.counter+self.batch_size]
        label = self.label[self.counter:self.counter+self.batch_size]
        for i in tmp:
            i = i.reshape([28,28,1])/127.5 - 1.0
            next_batch.append(i)
        self.counter += self.batch_size
        return (np.array(next_batch), np.array(label), 0)

    def data2pic(self, sample):
        sample = (sample.reshape([sample.shape[1],sample.shape[2]]) + 1.0) * 127.5
        sample[sample<0.0] = 0.0
        sample[sample>255.0] = 255.0
        return Image.fromarray(sample, mode='L')

    def generate_z(self, z_dim=100):
        sample = np.random.normal(0.,1.0,size=[self.batch_size, z_dim])
        #sample[sample>1.0] = 1.0
        #sample[sample<-1.0] = -1.0
        return sample



class real_data():
    def __init__(self, real_folder='./real_data/', batch_size=128):
        self.path = real_folder
        self.counter = 0
        self.batch_size = batch_size
        self.data = self.get_real_input()
        self.limit = self.data.shape[0]


    def get_next_batch(self):
        real_batch = []
        if (self.counter+self.batch_size)>=self.limit:
            self.counter = 0
            random.shuffle(self.data)
        images = self.data[self.counter:self.counter+self.batch_size]
        for i in images:
            img = Image.open(i).convert('L')
            img = np.asarray(img, dtype=np.float32).reshape([30,100,1])/127.5 - 1.0
            real_batch.append(img)
        self.counter += self.batch_size
        label = np.asarray([[0.,1.]]*self.batch_size)
        label[:,0] += np.random.uniform(0,0.1)
        label[:,1] += np.random.uniform(-0.1, 0.1)
        return (np.asarray(real_batch), label[:,1].reshape([self.batch_size,1]), label[:,0].reshape([self.batch_size,1]))


    def get_real_input(self):
        real_data = []
        for image_file in os.listdir(self.path):
            if image_file.endswith('.jpg'):
                real_data.append(os.path.join(self.path, image_file))
        random.shuffle(real_data)
        return np.asarray(real_data)


    def data2pic(self, sample):
        sample = (sample.reshape([sample.shape[1],sample.shape[2]]) + 1.0) * 127.5
        sample[sample<0.0] = 0.0
        sample[sample>255.0] = 255.0
        return Image.fromarray(sample, mode='L')

    def generate_z(self, z_dim=100):
        sample = np.random.normal(0.,1.0,size=[self.batch_size, z_dim])
        #sample[sample>1.0] = 1.0
        #sample[sample<-1.0] = -1.0
        return sample


'''
def get_next_batch(pointer, batch_size=75, data=real_data):
    real_batch = []
    images = data[pointer*batch_size:(pointer+1)*batch_size]
    for i in images:
        img = Image.open(i).convert('L')
        img = np.asarray(img, dtype=np.float32).reshape([100,30,1])
        real_batch.append(img)
    return (np.asarray(real_batch), np.asarray([[0,1]]*batch_size))

def get_next_batch_from_net(batch_size=75, url='http://pin.aliyun.com//get_img?sessionid=k0WDUmi7iD9YJ86CdXvLHhkZO5AYS9gKgG&identity=isvportal'):
    real_batch = []
    for i in xrange(batch_size):
        img = cStringIO.StringIO(urllib.urlopen(url).read())
        img = Image.open(img).convert('L')
        img = np.asarray(img, dtype=np.float32).reshape([100,30,1])
        real_batch.append(img)
    return (np.asarray(real_batch), np.asarray([0,1]*batch_size))
'''