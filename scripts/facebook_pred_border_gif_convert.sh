#!/usr/bin/env bash

process_dir='../examples/data/facebook_sempred/'

for val_batch_dir in `ls ${process_dir}`
do
#    echo ${val_batch_dir}
    for val_file in `ls ${process_dir}/${val_batch_dir}`
    do
#        echo ${val_file}
#        4,5,6是将来帧
        if [ ${val_file} == '4.png' ] || [ ${val_file} == '5.png' ] ||  [ ${val_file} == '6.png' ]; then
#            echo ${val_file}
            `convert -shave 2x2 -border 2x2 -bordercolor "#FF0000" ${process_dir}/${val_batch_dir}/${val_file} ${process_dir}/${val_batch_dir}/red_${val_file}`
        elif [ ${val_file} == '0.png' ] || [ ${val_file} == '1.png' ] ||  [ ${val_file} == '2.png' ] ||  [ ${val_file} == '3.png' ]; then
#            echo ${val_file}
            `cp ${process_dir}/${val_batch_dir}/${val_file} ${process_dir}/${val_batch_dir}/red_${val_file}`
        fi
    done
    convert -delay 20 -loop 0 ${process_dir}/${val_batch_dir}/red_*.png ${process_dir}/${val_batch_dir}/${val_batch_dir}.gif
done