#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
import torch
from torch.autograd import Variable

from videopred.dataloader.config import vpn_mnist_config
from videopred.dataloader.vpn_minst import GenerateMovingMnistData
from videopred.modelloader.prednet.prednet import PredNet

if __name__ == '__main__':
    data_generate = GenerateMovingMnistData()
    A_channels = (3, 48, 96, 192)
    R_channels = (3, 48, 96, 192)
    lr = 0.001 # if epoch < 75 else 0.0001
    num_epochs = 200000
    model = PredNet(R_channels, A_channels, output_mode='error')

    resume_model = 'training_700.pt'
    model.load_state_dict(torch.load(resume_model))
    start_epoch_id1 = resume_model.rfind('_')
    start_epoch_id2 = resume_model.rfind('.')
    start_epoch = int(resume_model[start_epoch_id1 + 1:start_epoch_id2]) + 1

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_scheduler(optimizer, epoch):
        if epoch < num_epochs // 2:
            return optimizer
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        return optimizer


    layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]))
    time_loss_weights = 1. / (vpn_mnist_config.num_timestamps- 1) * torch.ones(vpn_mnist_config.num_timestamps, 1)
    time_loss_weights[0] = 0
    time_loss_weights = Variable(time_loss_weights)

    for epoch in range(start_epoch, num_epochs, 1):
        optimizer = lr_scheduler(optimizer, epoch)
        warmup_batch, dataset_batch = data_generate.next_batch()
        dataset_batch = dataset_batch.transpose(0, 1, 4, 2, 3)
        #  print(dataset_batch.shape)
        dataset_batch_torch = torch.from_numpy(dataset_batch)
        dataset_batch_torch = Variable(dataset_batch_torch)
        #  print(dataset_batch_torch.shape)
        #  x = Variable(torch.randn(4, 10, 1, 120, 120))
        errors = model(dataset_batch_torch)  # batch x n_layers x nt
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, vpn_mnist_config.num_timestamps), time_loss_weights)  # batch*n_layers x 1
        errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
        errors = torch.mean(errors)

        optimizer.zero_grad()

        errors.backward()

        optimizer.step()
        # print(errors.data[0])
        if epoch%100 == 0:
            print(errors.data[0])
        if epoch%500 == 0:
            # print(errors.data[0])
            if epoch != 0:
                model_save_name = 'training_' + str(epoch) + '.pt'
                torch.save(model.state_dict(), model_save_name)
                print(model_save_name + ' is save')


