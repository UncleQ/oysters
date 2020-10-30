# coding=utf-8
import tushare as ts


def main():
    print(ts.__version__)
    token = None
    with open('../conf/tushare.token', 'r') as f:
        token = f.read()
    pro = ts.pro_api(token)
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
