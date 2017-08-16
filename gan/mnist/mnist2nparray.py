# -*- encoding:utf-8 -*-
import numpy as np
import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

if __name__=='__main__':
    train_x = read_idx('./train-images.idx3-ubyte')
    train_y = read_idx('./train-labels.idx1-ubyte')
    test_x = read_idx('./t10k-images.idx3-ubyte')
    test_y = read_idx('./t10k-labels.idx1-ubyte')
    all_x = np.concatenate((train_x, test_x), axis=0)
    all_y = np.concatenate((train_y, test_y), axis=0)

    np.save('train_x', train_x)
    np.save('train_y', train_y)
    np.save('test_x', test_x)
    np.save('test_y', test_y)
    np.save('all_x', all_x)
    np.save('all_y', all_y)
