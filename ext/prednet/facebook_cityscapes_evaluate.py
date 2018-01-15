# -*- coding: UTF-8 -*-
'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *


n_plot = 40
batch_size = 10
nt = 5

# 相关的weights，json的文件
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_facebook_segmpred_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_facebook_segmpred_model.json')
# weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
# json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
# weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights-extrapfinetuned.hdf5')  # where weights will be saved
# json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model-extrapfinetuned.json')
test_file = os.path.join(DATA_DIR, 'facebook_segmpred_X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'facebook_segmpred_sources_test.hkl')

# Load trained model
# 加载模型的json文件
f = open(json_file, 'r')
# 读取的json文件
json_string = f.read()
f.close()
# 从训练后存储的模型中序列化出模型，同时包含PredNet模型定制的参数，之后加载权重模型
# 存储模型将相应的json文件和weights文件存储即可，加载模型从对应的json文件和weights文件反序列化即可
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
# 创建测试模型
# 训练模型包含了InputLayer，PredNet等等，这里选取第二层即为PredNet
# print(train_model.layers)
layer_config = train_model.layers[1].get_config()
# 评估版本中将output_mode输出模型从误差error修改为predication预测
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
# 将网络中部分修改参数加载重构为PredNet网络，keras中具有get_config和get_weights等方法
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
# 输入层的shape为不包括batch的batch_input_shape从第一列之后的所有
# input_shape = list(train_model.layers[0].batch_input_shape[1:])
# 输入数据为nt，总共有10帧，来预测将来的一帧
# input_shape[0] = nt
# print('input_shape:', input_shape)
test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()
input_shape = X_test.shape[1:]
# print('input_shape:', input_shape)
# 构建输入层
inputs = Input(shape=tuple(input_shape))
# 将输入层输入到prednet网络中测试输出
predictions = test_prednet(inputs)
# 构建输入和输出模型
test_model = Model(inputs=inputs, outputs=predictions)

# 测试评估数据生成器
# test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
# X_test = test_generator.create_all()
# 预测模型时参照batch_size，一个批次的进行load然后predict
X_hat = test_model.predict(X_test, batch_size)
# 这里模型的默认通道均在最后一位
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
print('X_hat.shape:', X_hat.shape)
print('X_test.shape:', X_test.shape)
# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
# 比较测试结果
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
    plt.clf()
