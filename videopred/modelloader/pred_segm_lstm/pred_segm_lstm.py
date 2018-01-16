#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable


from ConvLSTMCell import ConvLSTMCell

class ConvReLU(nn.Module):
    def forward(self, x):
        x = self.conv(x)
        out = self.relu(x)
        return out

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvReLU, self).__init__()
        # pytorch中未实现padding=SAME因此这里设置stride=1然后padding补充
        padding=(kernel_size-1)/2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        pass

class TransConvReLU(nn.Module):
    def forward(self, x):
        x = self.trans_conv(x)
        out = self.relu(x)
        return out

    def __init__(self, in_channels, out_channels, kernel_size):
        super(TransConvReLU, self).__init__()
        padding=(kernel_size-1)/2
        self.trans_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        pass


class BasicPredNet(nn.Module):
    def __init__(self, in_channels):
        super(BasicPredNet, self).__init__()
        self.conv1 = ConvReLU(in_channels=in_channels, out_channels=8, kernel_size=3)
        self.conv2 = ConvReLU(in_channels=8, out_channels=8, kernel_size=3)
        self.conv3 = ConvReLU(in_channels=8, out_channels=8, kernel_size=3)
        self.conv4 = ConvReLU(in_channels=8, out_channels=4, kernel_size=1)
        # not implement
        # self.lstm = ConvReLU(in_channels=4, out_channels=8, kernel_size=3)
        self.lstm = ConvLSTMCell(4, 8)
        self.trans_conv5 = TransConvReLU(in_channels=8, out_channels=8, kernel_size=1)
        self.trans_conv6 = TransConvReLU(in_channels=8, out_channels=8, kernel_size=3)
        self.trans_conv7 = TransConvReLU(in_channels=8, out_channels=8, kernel_size=3)
        self.trans_conv8 = TransConvReLU(in_channels=8, out_channels=3, kernel_size=3)
        pass

    def forward(self, x, hidden):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # print('self.conv4_out:', x.size())

        x_hidden, x_cell = self.lstm(x, hidden)

        # print('x_hidden:', x_hidden.size())
        # print('x_cell:', x_cell.size())

        x = self.trans_conv5(x_cell)
        x = self.trans_conv6(x)
        x = self.trans_conv7(x)
        out = self.trans_conv8(x)
        return out, x_hidden


if __name__ == '__main__':
    num_sequence = 10
    hidden = None
    x = Variable(torch.randn(1, num_sequence, 3, 56, 56))
    for id_sequence in range(num_sequence):
        x_id = x[:, id_sequence, :, :, :]
        net = BasicPredNet(in_channels=3)
        pred_out, pred_hidden = net(x_id, hidden)
        hidden = pred_hidden
        print('hidden:', hidden.data.shape)

    # hidden = None
    # net = BasicPredNet(in_channels=3)
    # pred_out, pred_hidden = net(x, hidden)
    # print(pred_out.data.shape)
    pass

