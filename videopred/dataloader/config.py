from __future__ import absolute_import

class moving_mnist_config:
    def __init__(self):
        pass
    moving_mnist_data = '/home/cgf/Data/pytorch/train-images-idx3-ubyte.gz'


class vpn_mnist_config:
    def __init__(self):
        pass
    data_dir = '/home/cgf/Data/mnist_test_seq.npy'
    train_sequences_num = 7000
    truncated_steps = 9
    batch_size = 1
