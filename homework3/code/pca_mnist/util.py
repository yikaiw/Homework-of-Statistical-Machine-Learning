import gzip
import numpy as np
import os

def _load_img(file_path, img_size):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, img_size)

def _load_label(file_path):
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def load_mnist():
    path_list = {
        'train_img': r'data/train-images-idx3-ubyte.gz',
        'train_label': r'data/train-labels-idx1-ubyte.gz',
        'val_img': r'data/t10k-images-idx3-ubyte.gz',
        'val_label': r'data/t10k-labels-idx1-ubyte.gz'
    }
    img_dim = 784
    train_x = _load_img(path_list['train_img'], img_dim)
    train_y = _load_label(path_list['train_label'])
    val_x = _load_img(path_list['val_img'], img_dim)
    val_y = _load_label(path_list['val_label'])
    return train_x, train_y, val_x, val_y