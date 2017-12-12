#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from .prednet_convlstmcell import ConvLSTMCell

class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels, output_mode='error'):
        super(PredNet, self).__init__()
        # 最后一层没有表示单元，使用0代替
        self.r_channels = R_channels + (0,)
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        self.output_mode = output_mode

        # 训练时输出模式为error，测试时输出模型为prediction
        default_output_modes = ['prediction', 'error']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        # prednet网络的层数，包括误差项和上一时刻的表示项，而误差项由于包含正向和反向激活，所以大小为原始大小的2倍
        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i + 1], self.r_channels[i], (3, 3))
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(
                nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1),
                nn.ReLU(),
            )
            # 在第一层中增加StaLU饱和LU防止数据过大
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)

        # prednet中的上采样和最大池化
        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 输入的更新和下一层的A网络层大小相同
        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(
                nn.Conv2d(2 * self.a_channels[l], self.a_channels[l + 1], (3, 3), padding=1),
                self.maxpool
            )
            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, x):
        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        # 输入为bath_size*time_stamp*height*width
        w, h = x.size(-1), x.size(-2)
        batch_size = x.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(batch_size, 2*self.a_channels[l], w, h))
            R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h))
            w = w // 2
            h = h // 2
        time_steps = x.size(1)
        total_error = []

        # 输入的时间步骤进行下一帧的预测
        for t in range(time_steps):
            A = x[:, t]
            A = A.type(torch.FloatTensor)

            # 首先进行自顶向下的更新
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                # 初始时刻
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                # 最后一层，获取状态，其他几层将后面的层上采样进行连接
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    # 因为这里E的大小是原先的两倍，所以需要上采样匹配shape
                    tmp = torch.cat((E, self.upsample(R_seq[l + 1])), 1)
                    R, hx = cell(tmp, hx)
                R_seq[l] = R
                H_seq[l] = hx

            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    frame_prediction = A_hat
                # CRELU激活，将正向和反向relu组合
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg], 1)
                E_seq[l] = E
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)
            if self.output_mode == 'error':
                mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
                # batch x n_layers
                total_error.append(mean_error)

        if self.output_mode == 'error':
            return torch.stack(total_error, 2)  # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            return frame_prediction


# 限制在0-255范围内，如果输入大于255赋值为255，如果输入小于0赋值为0
class SatLU(nn.Module):
    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, x):
        return F.hardtanh(x, self.lower, self.upper, self.inplace)

def main():
    A_channels = (3, 48, 96, 192)
    R_channels = (3, 48, 96, 192)
    x = Variable(torch.randn(1, 10, 3, 120, 120))
    model = PredNet(R_channels, A_channels, output_mode='error')
    model(x)

if __name__ == '__main__':
    main()

