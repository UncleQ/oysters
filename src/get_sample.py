# coding=utf-8
import pandas as pd
from scipy.signal import argrelextrema
import numpy as np
from matplotlib import pyplot as plt
import os


def find(path, name):
    file_full_path = os.path.join(path, name)
    train_days = 60
    pre_days = 20
    pic_index = 0
    stock_df = pd.read_csv(file_full_path, low_memory=False)
    close_series = stock_df['close']
    date_list = stock_df['trade_date'].to_list()
    close_list = stock_df['close'].to_list()
    l = len(stock_df)
    if l < train_days + pre_days:
        return
    index_list = list()
    index_test_list = list()
    for i in range(train_days):
        index_list.append(i)
    for i in range(train_days, train_days + pre_days):
        index_test_list.append(i)
    for i in range(l - (train_days + pre_days)):
        print('code-%s, index-%d' % (name, i))
        pos0 = i
        pos1 = i + train_days
        pos2 = i + train_days + pre_days
        # print('index=%d, close-150=%f, close-100=%f' % (i, close_list[pos2-1], close_list[pos1-1]))
        max_value = close_series[pos0:pos1].max()
        min_value = close_series[pos0:pos1].min()
        pre_min_value = close_series[pos1:pos2].min()
        mid_value = (max_value + min_value) / 2
        half_value = max_value - mid_value
        diff_value = mid_value * 0.2
        if half_value < diff_value and close_list[pos1-1] == max_value \
                and ((close_list[pos2-1] - close_list[pos1-1]) > (mid_value * 1.5)) \
                and pre_min_value >= close_list[pos1-1]:
            print('pos=%d, max=%f, min=%f' % (i, max_value, min_value))
            plt.plot(index_list, close_list[pos0:pos1], color='green', label='Test Price')
            plt.plot(index_test_list, close_list[pos1:pos2], color='blue', label='Test Price')
            xticks = list()
            x_index_list = list()
            x_cnt = (train_days + pre_days) / 10 + 1
            for y in range(int(x_cnt)):
                x_index_list.append(y * 10)
                xticks.append(date_list[pos0 + y * 10])
            plt.xticks(x_index_list, xticks, rotation=25)
            plt.title('hello')
            plt.axhline(y=mid_value, ls=":", c="red")
            # plt.show()
            pic_path = '/home/tars/data/pic/%s_%d.png' % (name, pic_index)
            pic_index += 1
            plt.savefig(pic_path)
            plt.cla()
            print(pic_path)
            # break

    pass


def main():
    for root, dirs, files in os.walk('/home/tars/data/new/'):
        for name in files:
            # print(name)
            find(root, name)


if __name__ == '__main__':
    main()
