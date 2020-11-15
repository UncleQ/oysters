#coding=utf-8
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
    # df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    # df = pro.stock_basic(exchange='', list_status='L')
    # df.to_csv("~/data/test.csv", encoding="utf-8", mode="w", header=True, index=False)
    one_path = '~/data/one/'
    base_info_file = '~/data/basic_info.csv'
    basic_df = pd.read_csv(base_info_file, low_memory=False)
    print(basic_df.shape)
    code_l = basic_df['ts_code']
    index = 0
    index_total = 0
    for i, v in code_l.items():
        ond_df = ts.pro_bar(ts_code=v, adj='hfq', start_date='20000101', end_date='20201106')
        full_path = os.path.join(one_path, v + '.csv')
        ond_df.to_csv(full_path, encoding="utf-8", mode="w", header=True, index=False)
        index += 1
        index_total += 1
        logging.info('get code-%s, index-%d, save-%s', v, index_total, full_path)
        if index == 5:
            # time.sleep(3)
            index = 0
    #df = ts.pro_bar(ts_code='000001.SZ', adj='qfq', start_date='20000101', end_date='20201106')
    #print(df)
    #print(df.shape)
    return
    df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001',
                       fields='exchange,cal_date,is_open,pretrade_date', is_open='0')
    print(df.shape)
    print(df)
    df = pro.daily(trade_date='20200325')
    print(df.shape)
    print(df)
    pass


if __name__ == '__main__':
    main()
