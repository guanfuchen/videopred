import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from videopred.modelloader.prednet.prednet import PredNet

num_epochs = 150
batch_size = 16
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
lr = 0.001 # if epoch < 75 else 0.0001
nt = 10 # num of time steps

# #  layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cuda())
# layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]))
# time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1)
# time_loss_weights[0] = 0
# #  time_loss_weights = Variable(time_loss_weights.cuda())
# time_loss_weights = Variable(time_loss_weights)

model = PredNet(R_channels, A_channels, output_mode='error')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    print(epoch)
