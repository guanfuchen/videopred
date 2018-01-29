#!/usr/bin/python
# -*- coding: UTF-8 -*-
import glob
import os
import numpy as np
from scipy import misc
import hickle as hkl

if __name__ == '__main__':
    data_prefix = 'facebook_segmpred_org_'
    sources_train = []
    X_train = []
    sources_val = []
    X_val = []
    # 间隔一定时间进行预测，防止间隔太少相似性过大
    frame_interval = 3
    HOME_PATH = os.path.expanduser('~')
    # valid_dirs = glob.glob(os.path.join(HOME_PATH, 'GitHub/Quick/HPEC/可行区域识别/工大拍摄原始数据/DJI_000[0-9]_160x128'))
    valid_dirs = glob.glob(os.path.join(HOME_PATH, 'Data/facebook_segmpred_Data/train_batch_*_org'))
    train_rate = 0.7
    val_rate = 1 - train_rate
    # print(valid_dirs)
    for valid_dir in valid_dirs:
        # 目录名
        valid_dir_name = valid_dir[valid_dir.rfind('/')+1:]
        print(valid_dir)
        print(valid_dir_name)
        valid_image_files = glob.glob(os.path.join(valid_dir, '*.png'))
        valid_image_files.sort()
        valid_image_files_len = len(valid_image_files)

        split_index = int(valid_image_files_len*train_rate)
        train_image_files = valid_image_files[:split_index]
        val_image_files = valid_image_files[split_index:]

        train_image_files = train_image_files[::frame_interval]
        val_image_files = val_image_files[::frame_interval]

        for train_image_file in train_image_files:
            print(train_image_file)
            train_image_data = misc.imread(train_image_file)
            # print(train_image_data.shape)
            X_train.append(train_image_data)
            sources_train.append(valid_dir_name)
        for val_image_file in val_image_files:
            print(val_image_file)
            val_image_data = misc.imread(val_image_file)
            # print(val_image_data.shape)
            X_val.append(val_image_data)
            sources_val.append(valid_dir_name)

    X_train = np.array(X_train)
    print('X_train.shape:', X_train.shape)
    print('len(sources_train):', len(sources_train))
    X_val = np.array(X_val)
    print('X_val.shape:', X_val.shape)
    print('len(sources_val):', len(sources_val))
    hkl.dump(X_train, data_prefix + 'X_train.hkl')
    hkl.dump(sources_train, data_prefix + 'sources_train.hkl')
    hkl.dump(X_val, data_prefix + 'X_val.hkl')
    hkl.dump(sources_val, data_prefix + 'sources_val.hkl')
