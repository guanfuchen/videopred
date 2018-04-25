# coding=utf-8
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
        self.batch_size = vpn_mnist_config.batch_size
        self.seq_len = sequences.shape[0]
        self.train_len = vpn_mnist_config.train_sequences_num
        assert self.train_len < self.seq_len
        self.test_len = self.seq_len - self.train_len

        self.train_sequences = sequences[:self.train_len]
        self.test_sequences = sequences[self.train_len:]

    def next_batch(self):
        while True:
            idx = np.random.choice(self.train_len, self.batch_size)
            current_sequence = self.train_sequences[idx]

            # 输入前面10帧，以及预测的后面10帧
            return current_sequence[:, 0:self.num_timestamps]

    def next_train_batch_keras(self):
        while True:
            for i in range(0, self.train_len, self.batch_size):
                for j in range(i, i+self.batch_size):
                    idx = np.random.choice(self.train_len, self.batch_size)
                    current_sequence = self.train_sequences[idx]
                    current_sequence = current_sequence / 255.0
                # 输入前面10帧，以及预测的后面10帧
                prev_data = current_sequence[:, 0:self.num_timestamps]
                future_data = current_sequence[:, 1:self.num_timestamps+1]
                yield prev_data, future_data

    def next_test_batch_keras(self):
        while True:
            for i in range(0, self.test_len, self.batch_size):
                for j in range(i, i+self.batch_size):
                    idx = np.random.choice(self.test_len, self.batch_size)
                    current_sequence = self.test_sequences[idx]
                    current_sequence = current_sequence / 255.0
                # 输入前面10帧，以及预测的后面10帧
                prev_data = current_sequence[:, 0:self.num_timestamps]
                future_data = current_sequence[:, 1:self.num_timestamps+1]
                yield prev_data, future_data

    def test_batch(self):
        while True:
            idx = np.random.choice(self.test_sequences.shape[0], self.batch_size)
            current_sequence = self.test_sequences[idx]

            # 输入前面10帧，以及预测的后面10帧
            return current_sequence[:, 0:self.num_timestamps]

if __name__ == '__main__':
    data_generate = GenerateMovingMnistData()
    train_batch = data_generate.next_batch()
    print(train_batch.shape)
