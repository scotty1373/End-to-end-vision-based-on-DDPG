#-*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import csv

# MANU = sys.argv[1]
# FILE_PATH = sys.argv[2]
FILE_PATH = 'train_log.txt'

def txt2list(_data_line):
    _data_line = _data_line.rstrip('\n').split(' ', 26)
    _data_line.pop(18)
    _frame_list = []
    _seq = [i for i in range(1, 26, 2)]
    for t in _seq:
        _frame_list.append(_data_line[t])
    return _frame_list

def list2csv():
    

if __name__ == '__main__':
    fp = open(FILE_PATH, "r")
    # 临时变量
    data_lens_temp = fp.readlines()
    data_lens = len(data_lens_temp)
    # 内存释放
    del data_lens_temp
    fp.seek(0, 0)
    # 参数初始化s
    df_list = np.zeros(shape=(data_lens, 13))
    for _i in range(data_lens):
        data_pre_line = fp.readline()
        df_list[_i] = txt2list(data_pre_line)
    df_dataset = pd.DataFrame(df_list)
    df_dataset.columns = ['EPISODE', 'TIMESTAMP', 'EPISODE_LENGTH', 'ACTION',
                    'REWARD', 'Avg_REWARD', 'training_Loss', 'Q_MAX',
                    'gap', 'v_ego', 'v_lead', 'time', 'a_ego']
    df_dataset.to_csv('train_data.csv')
    fp.close()

