# coding=utf-8
from __future__ import division
import argparse, pdb, os, numpy, imp
from datetime import datetime
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import models, utils


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='poke', help='breakout | seaquest | flappy | poke | driving')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-model', type=str, default='baseline-3layer')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-nfeature', type=int, default=64, help='number of feature maps in convnet')
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
# 一次周期训练500个
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-datapath', type=str, default='./data/', help='data folder')
parser.add_argument('-save_dir', type=str, default='./results/', help='where to save the models')
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.set_default_tensor_type('torch.FloatTensor')

# if opt.gpu > 0:
#     torch.cuda.set_device(opt.gpu)


# load data and get dataset-specific parameters
# 数据加载和数据集相关的参数，这里以poke为例，读取config.json中poke相关的json数据
data_config = utils.read_config('config.json').get(opt.task)
# 数据集的batchsize
data_config['batchsize'] = opt.batch_size
# 数据集的路径，在当前路径的相对data/poke文件夹中
data_config['datapath'] = '{}/{}'.format(opt.datapath, data_config['datapath'])
# 条件个数，先前帧的个数
opt.ncond = data_config['ncond']
# 视频预测的个数
opt.npred = data_config['npred']
# 视频的高度和宽度
opt.height = data_config['height']
opt.width = data_config['width']
# nc代表
opt.nc = data_config['nc']
# 图像加载器，这里以poke为例
ImageLoader=imp.load_source('ImageLoader', 'dataloaders/{}.py'.format(data_config.get('dataloader'))).ImageLoader
# 图像加载初始化，输入为配置文件的json数据
dataloader = ImageLoader(data_config)


# Set filename based on parameters
# 基于参数相关，比如任务等创建区别的存储路径，以poke为例子，这里储存在./results/poke文件夹下，同时模型文件的名字为model=baseline-3layer模型类型，
# loss=l2视频loss，ncond=1视频先前帧数量，npred=1视频预测数量，nfeature=64特征数量，lrt=0.0005学习率
opt.save_dir = '{}/{}/'.format(opt.save_dir, opt.task)
opt.model_filename = '{}/model={}-loss={}-ncond={}-npred={}-nf={}-lrt={}'.format(
                    opt.save_dir, opt.model, opt.loss, opt.ncond, opt.npred, opt.nfeature, opt.lrt)
print("Saving to " + opt.model_filename)




############
### train ##
############
# 训练步骤，迭代次数nsteps
def train_epoch(nsteps):
    # 总共的loss，一个周期下nsteps=500的平均loss
    total_loss = 0
    # 设置模型训练模式，model.eval()设置模型评估模式
    model.train()
    for iter in range(0, nsteps):
        # 优化器和模型首先设置梯度为0，然后反向传播后，step一步
        optimizer.zero_grad()
        model.zero_grad()
        # 获取数据train
        cond, target, _ = dataloader.get_batch('train')
        # 视频概率数据和预测数据
        vcond = Variable(cond)
        vtarget = Variable(target)
        # forward
        # 模型预测结果
        pred = model(vcond)
        # 计算相应的loss
        loss = criterion(pred, vtarget)
        # 增加每一步的loss设置为总loss
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return total_loss / nsteps

# 测试步骤，迭代次数
def test_epoch(nsteps):
    total_loss = 0
    model.eval()
    for iter in range(0, nsteps):
        cond, target, action = dataloader.get_batch('valid')
        vcond = Variable(cond)
        vtarget = Variable(target)
        pred = model(vcond)
        loss = criterion(pred, vtarget)
        total_loss += loss.data[0]
    return total_loss / nsteps

# 训练
def train(n_epochs):
    # prepare for saving
    # 创建存储路径
    os.system("mkdir -p " + opt.save_dir)
    # training
    # 训练相关的loss
    best_valid_loss = 1e6
    # train_loss保存训练的loss，valid_loss保存校验的loss
    train_loss, valid_loss = [], []
    for i in range(0, n_epochs):
        print('epoch:', i)
        # 训练loss train_loss记录训练500次后的loss
        # 训练valid valid_loss记录训练500次后的loss
        train_loss.append(train_epoch(opt.epoch_size))
        valid_loss.append(test_epoch(opt.epoch_size))

        # 仅仅当valid_loss最后一次几乎达到最好的loss （1e6 初始值比较大）才存储并更新最好的loss为最后的校准loss
        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            # save 
            model.intype("cpu")
            # 存储周期，model模型，train_loss训练loss，valid_loss校准loss到模型文件中
            torch.save({ 'epoch': i, 'model': model, 'train_loss': train_loss, 'valid_loss': valid_loss},
                       opt.model_filename + '.model')
            # 同时存储优化器
            torch.save(optimizer, opt.model_filename + '.optim')
            # model.intype("gpu")

        # 一次周期 500次训练结束后打印log日志，其中包括迭代次数（周期数*没周期运行次数），train_loss训练loss，valid_loss校准loss，以及当前最好的best_valid_loss和学习率
        log_string = 'iter: {:d}, train_loss: {:0.6f}, valid_loss: {:0.6f}, best_valid_loss: {:0.6f}, lr: {:0.5f}'.format(
                      (i+1)*opt.epoch_size, train_loss[-1], valid_loss[-1], best_valid_loss, opt.lrt)
        print(log_string)
        utils.log(opt.model_filename + '.log', log_string)


if __name__ == '__main__':
    # 设置numpy和torch的随机种子，保证每次测试时第一次获取的随机数相同
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # build the model
    # 构建视频预测模型，设置输入n_in和输出n_out
    opt.n_in = opt.ncond * opt.nc
    opt.n_out = opt.npred * opt.nc
    # model = models.BaselineModel3Layer(opt).cuda()
    # 构建BaselineModel3Layer模型，参数按照opt
    model = models.BaselineModel3Layer(opt)
    # 使用Adam优化模型的参数
    optimizer = optim.Adam(model.parameters(), opt.lrt)
    if opt.loss == 'l1':
        # criterion = nn.L1Loss().cuda()
        # 评估loss
        criterion = nn.L1Loss()
    elif opt.loss == 'l2':
        # criterion = nn.MSELoss().cuda()
        # MSELoss
        criterion = nn.MSELoss()
    print('training...')
    # 打印训练日志，其中具有当前时间等等
    utils.log(opt.model_filename + '.log', '[training]')
    # 训练周期500次
    train(500)
