#coding=utf-8
import pandas as pd


def main():
    stock_df = pd.read_csv('data/new/000001.SZ.csv', low_memory=False)
    total_len = len(stock_df)
    print(total_len)
    pass


if __name__ == '__main__':
    main()

