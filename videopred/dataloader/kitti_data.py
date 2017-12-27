#!/usr/bin/python
# -*- coding: UTF-8 -*-

import hickle as hkl
from torch.utils.data import Dataset, DataLoader
from .config import kitti_data_config
import matplotlib.pyplot as plt



# Kitti数据接口
class KittiData(Dataset):
    def __init__(self, data_filename, source_filename, num_timestamps):
        super(KittiData, self).__init__()
        self.data_filename = data_filename
        self.source_filename = source_filename
        self.num_timestamps = num_timestamps
        self.data = hkl.load(data_filename)
        self.source = hkl.load(source_filename)
        self.start_index = []
        cur_index = 0
        # 将相同场景的视频帧合并在一起
        while cur_index < len(self.source) - self.num_timestamps + 1:
            if self.source[cur_index] == self.source[cur_index+self.num_timestamps-1]:
                self.start_index.append(cur_index)
                cur_index += self.num_timestamps
            else:
                cur_index += 1

    def __getitem__(self, index):
        start_loc = self.start_index[index]
        end_loc = start_loc + self.num_timestamps
        return self.data[start_loc:end_loc]

    def __len__(self):
        return len(self.start_index)

def main():
    kitti_val_dataset = KittiData(kitti_data_config.val_data_dir, kitti_data_config.val_source_dir, kitti_data_config.num_timestamps)
    print(len(kitti_val_dataset))
    print(kitti_val_dataset[0].shape)
    kitti_val_loader = DataLoader(kitti_val_dataset, batch_size=kitti_data_config.batch_size, shuffle=True)
    for kitti_index, kitti_val in enumerate(kitti_val_loader):
        # print kitti_index
        # print(kitti_val[0][0].numpy())
        plt.subplot(2, kitti_data_config.batch_size, kitti_index+1)
        plt.imshow(kitti_val[0][0].numpy())
    plt.subplot(2, kitti_data_config.batch_size, kitti_index+2)
    plt.imshow(kitti_val_dataset[0][0])
    plt.show()


if __name__ == '__main__':
    main()
