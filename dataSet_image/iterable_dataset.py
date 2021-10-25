# -*- coding: utf-8 -*-
import os
import re
import shutil

dataset_a = 'data_1630494341'
dataset_b = 'data_1631520771'
common_path = 'train'
train_path = 'images'
label_path = 'labels'
yolo_great = 'train.txt'


# 提供两个路径必须是相对于此文件存在目录的相对目录，生成yolo所需数据集结构
# └── dataSet_image
#     ├── data_1630494341
#     ├── data_1631520771
#     └── iterable_dataset.py
def dataset_convert(data_path_a, data_path_b):
    root_path = os.getcwd().replace('\\', '/')
    if not os.path.exists('./' + train_path):
        os.makedirs('./' + train_path + '/' + common_path)
    if not os.path.exists('./' + label_path):
        os.makedirs('./' + label_path + '/' + common_path)
    # file_list_a = os.listdir('./' + dataset_a)
    # file_list_b = os.listdir('./' + dataset_b)

    pattern = re.compile('^[0-9]*\.txt$')

    # train_dataset_anchor = open(yolo_great, 'w')
    counter = 0
    for dir_list in [os.path.join(root_path, data_path_a), os.path.join(root_path, data_path_b)]:
        os.chdir(dir_list)
        for file in os.listdir(dir_list):
            if re.match(pattern, file):
                temp_path = os.path.join(root_path, label_path, common_path, f'{counter:05}' + os.path.splitext(file)[1]).replace('\\', '/')
                shutil.copy(file, temp_path)
                print(f'get file name: {file}, copy to {temp_path}')
                temp_path = os.path.join(root_path, train_path, common_path, f'{counter:05}' + '.jpg').replace('\\', '/')
                shutil.copy(os.path.splitext(file)[0] + '.jpg', temp_path)
                print(f'get file name: {file}, copy to {temp_path}')
                counter += 1
            else:
                continue

    print(f'dataset split complete, sample num: {counter}')


if __name__ == '__main__':
    dataset_convert(dataset_a, dataset_b)

