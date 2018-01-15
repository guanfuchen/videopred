#!/usr/bin/python
# -*- coding: UTF-8 -*-
import glob
import os
import numpy as np
from scipy import misc
import hickle as hkl
import torchfile

# 第一个颜色为背景颜色，最后的值为0-19
colors = [[0, 0, 0],
          [128, 64, 128],
          [244, 35, 232],
          [70, 70, 70],
          [102, 102, 156],
          [190, 153, 153],
          [153, 153, 153],
          [250, 170, 30],
          [220, 220, 0],
          [107, 142, 35],
          [152, 251, 152],
          [0, 130, 180],
          [220, 20, 60],
          [255, 0, 0],
          [0, 0, 142],
          [0, 0, 70],
          [0, 60, 100],
          [0, 80, 100],
          [0, 0, 230],
          [119, 11, 32]]

label_colours = dict(zip(range(20), colors))

def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
#         print(l)
        r[temp == l] = label_colours[l+1][0]
        g[temp == l] = label_colours[l+1][1]
        b[temp == l] = label_colours[l+1][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

if __name__ == '__main__':
    data_prefix = 'facebook_segmpred_'
    save_dir = '{}_Data'.format(data_prefix)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # sources_train = []
    # X_train = []
    # sources_val = []
    # X_val = []
    # 间隔一定时间进行预测，防止间隔太少相似性过大
    frame_interval = 3
    HOME_PATH = os.path.expanduser('~')
    valid_dirs = glob.glob(os.path.join(HOME_PATH, 'Data/SegmPred/Data/*'))
    train_rate = 0.7
    val_rate = 1 - train_rate
    snap_first_train = False
    snap_first_val = False
    # print(valid_dirs)
    for valid_dir in valid_dirs:
        # 目录名
        valid_dir_name = valid_dir[valid_dir.rfind('/')+1:]
        # print(valid_dir)
        # print(valid_dir_name)

        if valid_dir_name == 'train':
            valid_image_files = glob.glob(os.path.join(valid_dir, '*.t7'))
            valid_image_files.sort()
            valid_image_files_len = len(valid_image_files)
            for valid_image_file in valid_image_files:
                valid_image_file_name = valid_image_file[valid_image_file.rfind('/') + 1:]
                # print('valid_image_file:', valid_image_file)
                # print('valid_image_file_name:', valid_image_file_name)
                SegmPredDataBatch = torchfile.load(valid_image_file)
                pred_sequences_segments = SegmPredDataBatch['R8s']
                # print(pred_sequences_segments.shape)
                for sequence_id in range(pred_sequences_segments.shape[0]):
                    # print('sequence_id:', sequence_id)
                    pred_sequences_segment = pred_sequences_segments[sequence_id]
                    # print('pred_sequences_segment.shape', pred_sequences_segment.shape)
                    for pred_segment_id, pred_segment in enumerate(pred_sequences_segment):
                        pred_segment = np.argmax(pred_segment, axis=0)
                        # print('pred_segment.shape', pred_segment.shape)
                        pred_segment_rgb = decode_segmap(pred_segment)
                        # print('pred_segment_rgb.shape', pred_segment_rgb.shape)
                        # X_train.append(pred_segment_rgb)


                        pred_segment_rgb_diff = 'train_{}_{}'.format(valid_image_file_name, sequence_id)
                        # print('pred_segment_rgb_diff', pred_segment_rgb_diff)
                        # sources_train.append(pred_segment_rgb_diff)
                        pred_segment_rgb_diff_save_dir = os.path.join(save_dir, pred_segment_rgb_diff)

                        if not os.path.exists(pred_segment_rgb_diff_save_dir):
                            os.mkdir(pred_segment_rgb_diff_save_dir)
                        misc.imsave(os.path.join(pred_segment_rgb_diff_save_dir, '{}.png'.format(pred_segment_id)), pred_segment_rgb)
                        # print('pred_segment_id:', pred_segment_id)

                        if not snap_first_train:
                            print('snap_first_train:', snap_first_train)
                            snap_first_train = True
                            misc.imsave('/tmp/tmp_train.png', pred_segment_rgb)
        elif valid_dir_name == 'val':
            valid_image_files = glob.glob(os.path.join(valid_dir, '*.t7'))
            valid_image_files.sort()
            valid_image_files_len = len(valid_image_files)
            for valid_image_file in valid_image_files:
                valid_image_file_name = valid_image_file[valid_image_file.rfind('/') + 1:]
                # print('valid_image_file:', valid_image_file)
                # print('valid_image_file_name:', valid_image_file_name)
                SegmPredDataBatch = torchfile.load(valid_image_file)
                pred_sequences_segments = SegmPredDataBatch['R8s']
                # print(pred_sequences_segments.shape)
                for sequence_id in range(pred_sequences_segments.shape[0]):
                    # print('sequence_id:', sequence_id)
                    # 校准数据是7个，4个输入值，3个输出值，同时大小不同，这里将数据修改为5位
                    # pred_sequences_segment = pred_sequences_segments[sequence_id][0:5]
                    pred_sequences_segment = pred_sequences_segments[sequence_id]
                    # print('pred_sequences_segment.shape', pred_sequences_segment.shape)
                    for pred_segment_id, pred_segment in enumerate(pred_sequences_segment):
                        pred_segment = np.argmax(pred_segment, axis=0)
                        # print('pred_segment.shape', pred_segment.shape)
                        pred_segment_rgb = decode_segmap(pred_segment)
                        # pred_segment_rgb = misc.imresize(pred_segment_rgb, (64, 64))
                        # print('pred_segment_rgb.shape', pred_segment_rgb.shape)
                        # X_val.append(pred_segment_rgb)

                        pred_segment_rgb_diff = 'val_{}_{}'.format(valid_image_file_name, sequence_id)
                        pred_segment_rgb_diff_save_dir = os.path.join(save_dir, pred_segment_rgb_diff)

                        if not os.path.exists(pred_segment_rgb_diff_save_dir):
                            os.mkdir(pred_segment_rgb_diff_save_dir)
                        misc.imsave(os.path.join(pred_segment_rgb_diff_save_dir, '{}.png'.format(pred_segment_id)), pred_segment_rgb)
                        # print('pred_segment_id:', pred_segment_id)

                        # print('pred_segment_rgb_diff', pred_segment_rgb_diff)
                        # sources_val.append(pred_segment_rgb_diff)

                        if not snap_first_val:
                            print('snap_first_val:', snap_first_val)
                            snap_first_val = True
                            misc.imsave('/tmp/tmp_val.png', pred_segment_rgb)

    # X_train = np.array(X_train)
    # print('X_train.shape:', X_train.shape)
    # print('len(sources_train):', len(sources_train))
    # X_val = np.array(X_val)
    # print('X_val.shape:', X_val.shape)
    # print('len(sources_val):', len(sources_val))
    # hkl.dump(X_train, data_prefix + 'X_train.hkl')
    # hkl.dump(sources_train, data_prefix + 'sources_train.hkl')
    # hkl.dump(X_val, data_prefix + 'X_val.hkl')
    # hkl.dump(sources_val, data_prefix + 'sources_val.hkl')
