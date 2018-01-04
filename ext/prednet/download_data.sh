#!/usr/bin/env bash
savedir="kitti_data"
mkdir -p -- "$savedir"
export http_proxy="http://127.0.0.1:8123"
export https_proxy="http://127.0.0.1:8123"
wget -c https://www.dropbox.com/s/rpwlnn6j39jjme4/kitti_data.zip?dl=0 -O $savedir/prednet_kitti_data.zip
unzip $savedir/prednet_kitti_data.zip -d $savedir
