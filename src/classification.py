# coding=utf-8
import pandas as pd
import numpy as np
import logging
import os


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',                       # 日志格式
                    datefmt='%Y-%m-%d %H:%M:%S')
                    # ,    # 时间格式：2018-11-12 23:50:21
                    # filename='/tmp/test.log',    # 日志的输出路径
                    # filemode='w')


def trans_df(file_name):
    old_path = '~/data/one'
    new_path = '~/data/new'
    stock_df = pd.read_csv(os.path.join(old_path, file_name), low_memory=False)
    series_open = stock_df['open']
    nan_pos = -1 
    for i,v in series_open.items():
        if pd.isna(v):
            nan_pos = i
            break
    if nan_pos != -1:
        stock_df = stock_df[:nan_pos]
        logging.info('%s NaN pos is %d' % (file_name, nan_pos))
    else:
        logging.info('%s has no NaN', file_name)
    stock_df = stock_df.iloc[::-1]
    stock_df.to_csv(os.path.join(new_path, file_name), encoding="utf-8", mode="w", header=True, index=False)


def main():
    print('hello')
    for root, dirs, files in os.walk('/home/tars/data/one'):
        for name in files:
            # print(name)
            trans_df(name)
            

if __name__ == '__main__':
    #trans_df('600213.SH.csv')
    #trans_df('000001.SZ.csv')
    main()

