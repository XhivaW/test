# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd 
import numpy as np 
import os
import random
from PIL import Image
import urllib
import cStringIO

def get_real_input(real_folder):
    real_data = []
    for image_file in os.listdir(real_folder):
        if image_file.endswith('.jpg'):
            real_data.append(os.path.join(real_folder, image_file))
    random.shuffle(real_data)
    return real_data

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

def generate_z(batch_size=75, z_dim=20):
    return np.random.uniform(-1.,1.,size=[batch_size, z_dim])
