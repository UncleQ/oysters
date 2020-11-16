# coding=utf-8
import pandas as pd
from matplotlib import pyplot as plt


def main():
    stock_df = pd.read_csv('/home/tars/data/new/000001.SZ.csv', low_memory=False)
    total_len = len(stock_df)
    print(total_len)
    # index_list = [0, 5, 10, 19, 40, 60, 90, 120, 240]
    index_list = list()
    for i in range(2001):
        index_list.append(i)
    print(len(index_list))
    close_list = stock_df['close'].tolist()
    close_list = close_list[:2000]
    close_list.append(1)
    plt.plot(index_list, close_list, color='green', label='Test Price')
    xticks = list()
    x_index_list = list()
    for i in range(11):
        x_index_list.append(i * 100)
        xticks.append(str(i * 100))
    plt.xticks(x_index_list, xticks, rotation=25)
    plt.title('hello')
    plt.show()
    pass


if __name__ == '__main__':
    main()

