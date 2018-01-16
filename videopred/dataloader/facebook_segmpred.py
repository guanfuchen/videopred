from __future__ import absolute_import

import numpy as np
import hickle as hkl

from videopred.dataloader.config import facebook_segmpred_config


class GenerateFaceBookSegmPredData:
    def __init__(self):
        np.random.seed(123)
        sequences = hkl.load(facebook_segmpred_config.data_dir)
        sequences = np.expand_dims(np.squeeze(sequences), 4)
        shuffled_idxs = np.arange(sequences.shape[0])
        np.random.shuffle(shuffled_idxs)
        sequences = sequences[shuffled_idxs]

        # print('sequences.shape:', sequences.shape)
        self.num_timestamps = sequences.shape[1]
        self.height = sequences.shape[2]
        self.width = sequences.shape[3]
        self.channel = sequences.shape[4]

        self.train_sequences = sequences[:facebook_segmpred_config.train_sequences_num]
        self.test_sequences = sequences[facebook_segmpred_config.train_sequences_num:]

    def next_batch(self):
        while True:
            idx = np.random.choice(self.train_sequences.shape[0], facebook_segmpred_config.batch_size)
            current_sequence = self.train_sequences[idx]

            # current_sequence = current_sequence / 19.0

            return current_sequence

    def test_batch(self):
        while True:
            idx = np.random.choice(self.test_sequences.shape[0], facebook_segmpred_config.batch_size)
            current_sequence = self.test_sequences[idx]

            # current_sequence = current_sequence / 19.0

            return current_sequence

if __name__ == '__main__':
    data_generate = GenerateFaceBookSegmPredData()
    train_batch = data_generate.next_batch()
    print(train_batch.shape)
