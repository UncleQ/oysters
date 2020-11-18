# coding=utf-8
import pandas as pd
from scipy.signal import argrelextrema
import numpy as np
from matplotlib import pyplot as plt
import os

# from tensorflow.contrib.learn.python.learn.models import linear_regression


def least_squares(x, y):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(50):
        k = (x[i]-x_) * (y[i]-y_)
        m += k
        p = np.square(x[i]-x_)
        n = n + p
    a = m/n
    b = y_ - a * x_
    return a, b


def linear_regression(x, y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    return np.linalg.solve(A, b)


def find(path, name, total_len, cur_index):
    file_full_path = os.path.join(path, name)
    train_days = 40
    pre_days = 40
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
        # print('%d/%d, code-%s, index-%d' % (cur_index, total_len, name, i))
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
        nx = np.array(index_test_list)
        ny = np.array(close_list[pos1:pos2])
        a0, a1 = linear_regression(nx, ny)
        if half_value < diff_value and close_list[pos1-1] == max_value and 0.58 < a1 < 1.73 \
                and pre_min_value >= close_list[pos1-1]:
                # and ((close_list[pos2-1] - close_list[pos1-1]) > (mid_value * 1.4)) \
            print('code-%s, pos=%d, max=%f, min=%f' % (name, i, max_value, min_value))
            plt.plot(index_list, close_list[pos0:pos1], color='green', label='Test Price')
            plt.plot(index_test_list, close_list[pos1:pos2], color='blue', label='Test Price')
            y1 = [a0 + a1 * x for x in index_test_list]
            plt.plot(index_test_list, y1, 'r-', lw=2, markersize=6)
            plt.plot(0, 0, '.y')
            xticks = list()
            x_index_list = list()
            x_cnt = (train_days + pre_days) / 10 + 1
            for y in range(int(x_cnt)):
                x_index_list.append(y * 10)
                xticks.append(date_list[pos0 + y * 10])
            plt.xticks(x_index_list, xticks, rotation=25)
            plt.title('hello')
            plt.axhline(y=mid_value, ls=":", c="red")
            plt.axhline(y=mid_value * 1.4, ls=":", c="blue")
            # plt.show()
            pic_path = '/home/tars/data/pic3/%s_%d.png' % (name, pic_index)
            pic_index += 1
            plt.savefig(pic_path)
            plt.cla()
            print(pic_path)
            # break

    pass


def main():
    for root, dirs, files in os.walk('/home/tars/data/new/'):
        l = len(files)
        i = 0
        for name in files:
            # print(name)
            i += 1
            print('%d/%d, code-%s' % (i, l, name))
            find(root, name, l, i)


if __name__ == '__main__':
    main()
