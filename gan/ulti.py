# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd 
import numpy as np 
import os
import random
from PIL import Image

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
        img = Image.open(i)
        img = np.asarray(img, dtype=float32).reshape([100,30,3])
        real_batch.append(img)
    return real_batch

