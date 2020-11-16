# coding=utf-8
import pandas as pd
from scipy.signal import argrelextrema
import numpy as np
from matplotlib import pyplot as plt
import os


def find(file_full_path):
    stock_df = pd.read_csv(file_full_path, low_memory=False)
    close_series = stock_df['close']
    close_list = stock_df['close'].to_list()
    l = len(stock_df)
    if l < 150:
        return
    index_list = list()
    index_test_list = list()
    for i in range(60):
        index_list.append(i)
    for i in range(60, 80):
        index_test_list.append(i)
    for i in range(l - 80):
        pos0 = i
        pos1 = i + 60
        pos2 = i + 80
        # print('index=%d, close-150=%f, close-100=%f' % (i, close_list[pos2-1], close_list[pos1-1]))
        max_value = close_series[pos0:pos1].max()
        min_value = close_series[pos0:pos1].min()
        pre_min_value = close_series[pos1:pos2].min()
        mid_value = (max_value + min_value) / 2
        half_value = max_value - mid_value
        diff_value = mid_value * 0.2
        if half_value < diff_value and ((close_list[pos2-1] - close_list[pos1-1]) > (mid_value * 1.5)) \
                and pre_min_value >= close_list[pos1-1]:
            print('pos=%d, max=%f, min=%f' % (i, max_value, min_value))
            plt.plot(index_list, close_list[pos0:pos1], color='green', label='Test Price')
            plt.plot(index_test_list, close_list[pos1:pos2], color='blue', label='Test Price')
            xticks = list()
            x_index_list = list()
            for y in range(9):
                x_index_list.append(y * 10)
                xticks.append(str(y * 10))
            plt.xticks(x_index_list, xticks, rotation=25)
            plt.title('hello')
            plt.axhline(y=mid_value, ls=":", c="red")
            plt.show()
            break

    pass


def main():
    for root, dirs, files in os.walk('/home/tars/data/new/'):
        for name in files:
            # print(name)
            find(os.path.join(root, name))


if __name__ == '__main__':
    main()
