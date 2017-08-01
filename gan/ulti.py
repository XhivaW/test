# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd 
import numpy as np 
import os
import random
from PIL import Image
import urllib
import cStringIO

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
            img = np.asarray(img, dtype=np.float32).reshape([100,30,1])
            real_batch.append(img)
        self.counter += self.batch_size
        return (np.asarray(real_batch), np.asarray([[0,1]]*self.batch_size))


    def get_real_input(self):
        real_data = []
        for image_file in os.listdir(self.path):
            if image_file.endswith('.jpg'):
                real_data.append(os.path.join(self.path, image_file))
        random.shuffle(real_data)
        return np.asarray(real_data)


    def data2pic(self, sample):
        sample = sample.reshape([30,100])
        return Image.fromarray(sample, mode='L')


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

def generate_z(batch_size=128, z_dim=20):
    return np.random.uniform(-1.,1.,size=[batch_size, z_dim])
