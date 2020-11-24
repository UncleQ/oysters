# coding=utf-8
import tushare as ts
import pandas as pd
import os
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',                       # 日志格式
                    datefmt='%Y-%m-%d %H:%M:%S')
                    # ,    # 时间格式：2018-11-12 23:50:21
                    # filename='/tmp/test.log',    # 日志的输出路径
                    # filemode='w')


def main():
    print(ts.__version__)
    token = None
    with open('../conf/tushare.token', 'r') as f:
        token = f.read()
    pro = ts.pro_api(token)
    daily_basic_path = '~/data/daily_basic/'
    base_info_file = '~/data/basic_info.csv'
    basic_df = pd.read_csv(base_info_file, low_memory=False)
    print(basic_df.shape)
    code_l = basic_df['ts_code']
    index = 0
    index_total = 0
    for i, v in code_l.items():
        index_total += 1
        if index_total <= 4002:
            continue
        df = pro.daily_basic(ts_code=v, start_date='20000101', end_date='20201106')
        full_path = os.path.join(daily_basic_path, v + '.csv')
        df.to_csv(full_path, encoding="utf-8", mode="w", header=True, index=False)
        logging.info('get code-%s, index-%d, save-%s', v, index_total, full_path)
        index += 1
        if index == 3:
            time.sleep(1)
            index = 0
    return


if __name__ == '__main__':
    main()
