#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
import os

from videopred.dataloader.vpn_minst import GenerateMovingMnistData

if __name__ == '__main__':

    data_generate = GenerateMovingMnistData()

    n_frames = data_generate.num_timestamps
    row = data_generate.height
    col = data_generate.width
    channel = data_generate.channel
    batch_size = data_generate.batch_size

    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=(n_frames, row, col, channel), padding='same',
                       return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernel_size=(1, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    callbacks = []
    model_checkpoint_filepath = "keras_conv_lstm_mnist-weights-best.hdf5"
    if os.path.exists(model_checkpoint_filepath):
        model.load_weights(model_checkpoint_filepath)
    model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks.append(model_checkpoint)

    # 增加val_data_generate评估数据集
    model.fit_generator(generator=data_generate.next_train_batch_keras(), validation_data=data_generate.next_test_batch_keras(), steps_per_epoch=data_generate.train_len / batch_size, validation_steps=data_generate.test_len / batch_size, epochs=1000, verbose=1, callbacks=callbacks)
    # model.fit_generator(generator=data_generate.next_train_batch_keras(), validation_data=data_generate.next_test_batch_keras(), steps_per_epoch=1, validation_steps=1, epochs=1000, verbose=1, callbacks=callbacks)

