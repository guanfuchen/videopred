#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import utils, imp, numpy
import matplotlib.pyplot as plt
# tester for dataloader
parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='poke', help='breakout | seaquest | flappy | poke | driving')
parser.add_argument('-datapath', type=str, default='./data')
opt = parser.parse_args()

# "poke": {
# "height": 240,
# "width": 240,
# "nc": 3, 
# "n_actions": 5, 
# "ncond": 1, 
# "npred": 1, 
# "phi_fc_size": 225, 
# "dataloader": "data_poke", 
# "datapath": "poke/data/"
# },

# 定义任务，然后从配置文件中读取data_config，
data_config = utils.read_config('config.json').get(opt.task)
data_config['datapath'] = '{}/{}'.format(opt.datapath, data_config['datapath'])
data_config['batchsize'] = 64
ImageLoader=imp.load_source('ImageLoader', 'dataloaders/{}.py'.format(data_config.get('dataloader'))).ImageLoader
dataloader = ImageLoader(data_config)

cond, target, action = dataloader.get_batch('test')

# show some images
N = 3
im=dataloader.plot_seq(cond[0:N].unsqueeze(1), target[0:N].unsqueeze(1))
plt.imshow(numpy.transpose(im.cpu().numpy(), (1, 2, 0)))
plt.show()

