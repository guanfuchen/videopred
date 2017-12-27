# videopred

---
## video prediction algorithms

这个仓库旨在实现常用的视频预测算法，主要参考如下：
- [ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch)
- [pytorch_convlstm](https://github.com/rogertrullo/pytorch_convlstm)
- ...

---
## 视频预测相关论文

[Prediction Under Uncertainty with Error-Encoding Networks](https://arxiv.org/abs/1711.04994) EEN

---
### 网络实现

- ConvLSTM
- PredNet [prednet实现](doc/prednet_implement.md)
- ...

---
### 数据集实现

- Moving MNIST
- UCF101
- ...

---
### 依赖

- pytorch
- ...

---
### 数据

- Moving MNIST
- Kitti数据集[kitti_dataset](doc/kitti_dataset.md)
- UCF101[ucf101_dataset](doc/ucf101_dataset.md)
- ...

---
### 用法

**可视化**

[visdom](https://github.com/facebookresearch/visdom)

```bash
# 在tmux或者另一个终端中开启可视化服务器visdom
python -m visdom.server
# 然后在浏览器中查看127.0.0.1:9097
```

**训练**
```bash
# 训练模型
python train.py
```

**校验**
```bash
# 校验模型
python validate.py
```

**测试**
```bash
# 测试模型
python test.py
```

