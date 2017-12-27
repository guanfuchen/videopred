# -*- coding: UTF-8 -*-
'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *


# 是否存储模型权重
save_model = True  # if weights will be saved
# 存储的权重文件
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

# Data files
# 数据文件，包括训练的文件和源以及校准的
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Training parameters
# 训练参数
nb_epoch = 150
batch_size = 1
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Model parameters
# 输入图像的维度为3,128,160，并且判断是否channels_first
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
# stack sizes为
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
# 滤波器A的大小，A_hax滤波器的大小，R滤波器的大小
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
# 每一层loss的权重，[1,0,0,0]表示L-0模型，[1,0.1,0.1,0.1]表示L-all模型
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
# 在训练期间输入的时间序列的timesteps，并且对于所有timesteps的数据都是相同的除了第一步
nt = 10  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)
# shape变为(nt,input_image_shape)
inputs = Input(shape=(nt,) + input_shape)
# 计算预测网络的误差，误差维度是batch_size，nt，nb_layers
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
# 通过层计算权重的误差
errors_by_time = TimeDistributed(Dense(1, weights=[layer_loss_weights, np.zeros(1)], trainable=False), trainable=False)(errors)  # calculate weighted error by layer
# 计算batch_size，nt的误差，级每一个时间的误差
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
# 对每一个误差进行权重赋值，这里取第一时间
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
# 构建预测模型，输入为时间轴，HxWxC，输出为final_errors，使用优化器adam，loss为MSE
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

# 训练数据生成器和校准数据生成器
train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

# 学习率lr在75周期后从0.001下降到0.0001
lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
# 如果存储模型则增加该模块
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

# 模型设置训练生成模块和校准生成模块，同时callbacks
history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

# 存储模型文件对应的string
if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
