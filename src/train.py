import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential  # Sequential 用于初始化神经网络
from keras.layers import Dense  # Dense 用于添加全连接的神经网络层
from keras.layers import LSTM  # LSTM 用于添加长短期内存层
from keras.layers import Dropout  # Dropout 用于添加防止过拟合的dropout层
from keras.layers import Activation
from keras import optimizers
import os
from optparse import OptionParser
from sklearn.metrics import mean_squared_error
from math import sqrt

from src.utils import data_utils

np.set_printoptions(suppress=True)


tag = 'p20_t100_e500_14_3_31'
batch_size = 128
epochs = 500
pre_day = 20
time_step = 100
divide_date = '2014/3/31'
divide_pos = 5000
test_range = -1
col_n = ['close',
         'turnover_rate',
         'turnover_rate_f',
         'volume_ratio',
         'pe',
         'pe_ttm',
         'pb',
         'ps',
         'ps_ttm',
         #'dv_ratio',
         #'dv_ttm',
         'total_share',
         'float_share',
         'free_share',
         'total_mv',
         'circ_mv']
# col_n = ['shoupanjia', 'zuigaojia', 'zuidijia', 'chengjiaojine', 'num']
columns = len(col_n)
data_base = pd.read_csv('~/data/daily_basic_new/000001.SZ.csv')
#data_base = pd.read_csv('../label_data_web.csv')
date_list = pd.DataFrame(data_base, columns=['trade_date']).values
#date_list = pd.DataFrame(data_base, columns=['date']).values
xs = [d[0] for d in date_list]
divide_pos = data_utils.get_divide_pos(xs, divide_date)
test_start = divide_pos + pre_day - 1
test_end = 0
if test_range == -1:
    test_end = -1
else:
    test_end = divide_pos + test_range + pre_day - 1
i_list = list()
d_list = list()


# 构建训练数据
def train_data():
    global tag, epochs, pre_day, time_step, divide_date, divide_pos, test_start, test_end, i_list, d_list
    dataset_train = pd.DataFrame(data_base, columns=col_n)
    # dataset_train = dataset_train.iloc[:, 2:]

    training_set = dataset_train.iloc[:, :].values
    # training_set = data_base.iloc[:, 2:12].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(time_step, divide_pos - pre_day):
        X_train.append(training_set_scaled[i - time_step:i, :])
        y_train.append(training_set_scaled[i + pre_day - 1, :])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], columns))
    return X_train, y_train, sc


# 构建测试数据
def test_data(sc):
    global tag, epochs, pre_day, time_step, divide_date, divide_pos, test_start, test_end, i_list, d_list
    dataset_train = pd.DataFrame(data_base, columns=col_n)
    dataset_total = dataset_train.iloc[:, :].values
    real_stock_price = dataset_total[test_start:test_end, 0]
    training_set_scaled = sc.fit_transform(dataset_total)
    #real_stock_price = training_set_scaled[5000 + pre_day - 1:5200 + pre_day - 1, 0]
    inputs = training_set_scaled
    X_test = []
    end_pos = 0
    if test_range != -1:
        end_pos = divide_pos + test_range
    else:
        end_pos = len(xs) - pre_day
    for i in range(test_start - pre_day, end_pos - 1):
        X_test.append(inputs[i - time_step:i, :])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], columns))

    return X_test, real_stock_price


# 创建股票预测模型
def stock_model(X_train, y_train):
    global tag, epochs, pre_day, time_step, divide_date, divide_pos, test_start, test_end, i_list, d_list
    regressor = Sequential()
    # LSTM的输入为 [samples, timesteps, features],这里的timesteps为步数，features为维度 这里我们的数据是6维的
    regressor.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], columns)))
    # regressor.add(Activation('relu'))  # 激活函数是tanh
    regressor.add(Dropout(0.2))
    # '''
    regressor.add(LSTM(units=128, return_sequences=True))
    # regressor.add(Activation('relu'))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=128, return_sequences=True))
    # regressor.add(Activation('relu'))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=128))
    # regressor.add(Activation('relu'))
    regressor.add(Dropout(0.2))
    
    # '''
    #regressor.add(Dense(500))  # 隐藏层节点500个
    #regressor.add(Activation('relu'))
    #regressor.add(Dropout(0.5))
    # 全连接，输出6个
    regressor.add(Dense(units=columns))
    #regressor.add(Activation('softmax'))

    #adam = optimizers.Adam(0.0006)
    adam = optimizers.Adam(0.0006)
    regressor.compile(optimizer=adam, loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return regressor


def main():
    global tag, epochs, pre_day, time_step, divide_date, divide_pos, test_start, test_end, i_list, d_list
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose")
    parser.add_option("-n", "--ndays", action="store", type="int", dest="ndays")
    parser.add_option("-e", "--epochs", action="store", type="int", dest="epochs")
    parser.add_option("-t", "--time_step", action="store", type="int", dest="time_step")
    parser.add_option("-o", "--out_base", action="store", type="string", dest="out_base")
    parser.add_option("-d", "--date", action="store", type="string", dest="test_date", default="foo.txt")
    (options, args) = parser.parse_args()
    epochs = options.epochs
    pre_day = options.ndays
    time_step = options.time_step
    divide_date = options.test_date
    out_base = options.out_base
    tag = 'n%d_t%d_e%d_%s' % (pre_day, time_step, epochs, data_utils.get_tag_date(divide_date))
    print(xs[-1])
    divide_pos = data_utils.get_divide_pos(xs, divide_date)
    print(divide_pos)
    print(xs[divide_pos])
    #test_start = divide_pos + 1
    #test_end = 0
    #if test_range == -1:
    #    test_end = -1
    #else:
    #    test_end = test_start + test_range + pre_day - 1
    #i_list, d_list = data_utils.get_month_date_list(xs[test_start:test_end])
    print(tag)
    #out_path = os.path.join(out_base, tag)
    print(out_base)

    X_train, y_train, sc = train_data()

    regressor = stock_model(X_train, y_train)
    model_path = os.path.join(out_base, tag)
    process_time = time.strftime("%Y%02m%02d_%02H%02M%02S", time.localtime())
    if os.path.isdir(model_path):
        back_path = os.path.join(out_base, '%s_%s' % (tag, process_time))
        os.rename(model_path, back_path)
    os.mkdir(model_path)
    regressor.save(os.path.join(model_path, 'model.h5'))
    #return
    X_test, real_stock_price = test_data(sc)
    predicted_stock_price = regressor.predict(X_test)
    scores = regressor.evaluate(X_test, predicted_stock_price, batch_size=200, verbose=0)
    print(scores)

    #predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    real_stock_price_list = real_stock_price.tolist()

    plt.plot(real_stock_price, color='black', label='Stock Price')
    # 显示开盘价
    origin_data = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price_list = origin_data[:, 0].tolist()
    sigma = sqrt(mean_squared_error(real_stock_price_list, predicted_stock_price_list))
    ab = np.array([real_stock_price_list, predicted_stock_price_list])
    corrocefab = np.corrcoef(ab)
    print(corrocefab)
    r = corrocefab[0, 1]
    plt.plot(origin_data[:, 0], color='green', label='Predicted Stock Price')
    d = int(len(i_list) / 15)
    test_len = len(xs[test_start:test_end])
    di = int(test_len / 10)
    i_list2 = list() 
    d_list2 = list()
    for i in range(10):
        i_list2.append(int(di * i)) 
        d_list2.append(xs[int(test_start + di * i)])
    plt.xticks(i_list2, d_list2, rotation=25)
    #plt.xticks(i_list[::d], d_list[::d], rotation=45)
    plt.title('r=%.2f%%,%s=%.1f,e=%s,ts=%d,n=%d,v=2.0' % (r * 100, chr(0x03c3), sigma, epochs, time_step, pre_day))
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    #plt.show()
    process_time = time.strftime("%Y%02m%02d_%02H%02M%02S", time.localtime())
    plt.savefig('SH_%s_ts%d_n%d_%s.png' % (data_utils.get_tag_date2(divide_date), time_step, pre_day, process_time))
    #plt.savefig('%s_%s.png' % (tag, process_time))
    #out_put_path = '/opt/tars/stock/result/%s_%s' % (tag, process_time)
    model_path = os.path.join(out_base, tag)
    if os.path.isdir(model_path):
        back_path = os.path.join(out_base, '%s_%s' % (tag, process_time))
        os.rename(model_path, back_path)
    os.mkdir(model_path)
    plt.savefig(os.path.join(model_path, 'show.png'))
    regressor.save(os.path.join(model_path, 'model.h5'))
    with open(os.path.join(model_path, 'info.txt'), 'w') as f:
        f.write('batch_size = %d\n' % batch_size)
        f.write('epochs = %d\n' % epochs)
        f.write('pre_day = %d\n' % pre_day)
        f.write('time_step = %d\n' % time_step)
        f.write('divide_date = %s\n' % divide_date)
        f.write('divide_pos = %d\n' % divide_pos)
        f.write('test_range = %d\n' % test_range)
        f.write('r = %f\n' % r)


if __name__ == '__main__':
    main()
