from __future__ import absolute_import

import numpy as np

from videopred.dataloader.config import vpn_mnist_config


class GenerateMovingMnistData:
    def __init__(self):
        np.random.seed(123)
        sequences = np.load(vpn_mnist_config.data_dir).transpose((1, 0, 2, 3))
        sequences = np.expand_dims(np.squeeze(sequences), 4)
        shuffled_idxs = np.arange(sequences.shape[0])
        np.random.shuffle(shuffled_idxs)
        sequences = sequences[shuffled_idxs]

        self.num_timestamps = 10
        self.height = sequences.shape[2]
        self.width = sequences.shape[3]
        self.channel = sequences.shape[4]

        self.train_sequences = sequences[:vpn_mnist_config.train_sequences_num]
        self.test_sequences = sequences[vpn_mnist_config.train_sequences_num:]

    def next_batch(self):
        while True:
            idx = np.random.choice(self.train_sequences.shape[0], vpn_mnist_config.batch_size)
            current_sequence = self.train_sequences[idx]

            return current_sequence[:, 0:self.num_timestamps]

    def test_batch(self):
        while True:
            idx = np.random.choice(self.test_sequences.shape[0], vpn_mnist_config.batch_size)
            current_sequence = self.test_sequences[idx]

            return current_sequence[:, 0:self.num_timestamps]

if __name__ == '__main__':
    data_generate = GenerateMovingMnistData()
    train_batch = data_generate.next_batch()
    print(train_batch.shape)
