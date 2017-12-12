from __future__ import absolute_import
import os

from os.path import expanduser
HOME_PATH = expanduser('~')

class moving_mnist_config:
    def __init__(self):
        pass
    moving_mnist_data = os.path.join(HOME_PATH, 'Data/pytorch/train-images-idx3-ubyte.gz')


class vpn_mnist_config:
    def __init__(self):
        pass
    data_dir = os.path.join(HOME_PATH, 'Data/mnist_test_seq.npy')
    train_sequences_num = 7000
    truncated_steps = 9
    batch_size = 4
    num_timestamps = 10


class kitti_data_config:
    def __init__(self):
        pass
    num_timestamps = 10
    batch_size = 4
    data_dir = os.path.join(HOME_PATH, 'Data/prednet/kitti_data')
    train_data_dir = os.path.join(data_dir, 'X_train.hkl')
    train_source_dir = os.path.join(data_dir, 'sources_train.hkl')
    test_data_dir = os.path.join(data_dir, 'X_test.hkl')
    test_source_dir = os.path.join(data_dir, 'sources_test.hkl')
    val_data_dir = os.path.join(data_dir, 'X_val.hkl')
    val_source_dir = os.path.join(data_dir, 'sources_val.hkl')
