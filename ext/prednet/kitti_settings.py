# -*- coding: UTF-8 -*-
# Where KITTI data will be saved if you run process_kitti.py
# If you directly download the processed data, change to the path of the data.
# 存储的KITTI数据集路径
DATA_DIR = './kitti_data/'

# Where model weights and config will be saved if you run kitti_train.py
# If you directly download the trained weights, change to appropriate path.
# 权重文件存储的路径并且在运行train的过程中也会存储在这个路径下
WEIGHTS_DIR = './model_data_keras2/'

# Where results (prediction plots and evaluation file) will be saved.
# 权重的结果(预测曲线和评估文件)存储路径
RESULTS_SAVE_DIR = './kitti_results/'
