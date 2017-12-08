from __future__ import absolute_import

import numpy as np
from .config import vpn_mnist_config
from .logger import Logger


class GenerateData:
    def __init__(self):
        np.random.seed(123)
        sequences = np.load(vpn_mnist_config.data_dir).transpose((1, 0, 2, 3))
        sequences = np.expand_dims(np.squeeze(sequences), 4)
        shuffled_idxs = np.arange(sequences.shape[0])
        np.random.shuffle(shuffled_idxs)
        sequences = sequences[shuffled_idxs]

        Logger.debug(('data shape', sequences.shape))

        self.train_sequences = sequences[:vpn_mnist_config.train_sequences_num]
        self.test_sequences = sequences[vpn_mnist_config.train_sequences_num:]

    def next_batch(self):
        while True:
            idx = np.random.choice(vpn_mnist_config.train_sequences_num, vpn_mnist_config.batch_size)
            current_sequence = self.train_sequences[idx]

            return current_sequence[:, :vpn_mnist_config.truncated_steps + 1], current_sequence[:, vpn_mnist_config.truncated_steps:2 * vpn_mnist_config.truncated_steps + 1]

    def test_batch(self):
        while True:
            idx = np.random.choice(self.test_sequences.shape[0], vpn_mnist_config.batch_size)
            current_sequence = self.test_sequences[idx]

            return current_sequence[:, :vpn_mnist_config.truncated_steps + 1], current_sequence[:, vpn_mnist_config.truncated_steps:2 * vpn_mnist_config.truncated_steps + 1]

if __name__ == '__main__':
    data_generate = GenerateData()
    warmup_batch, train_batch = data_generate.next_batch()
    print(warmup_batch.shape)
    print(train_batch.shape)
