"""
    This script loads the final LSTM models selected by grid search methods, prints their architecture
    prints their RMSE, MAE, and MAPE, then show the prediction results in the form of graphs.
"""

import os
from collections import deque
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# mae
from sklearn.metrics import mean_absolute_error as mae
# mape
from sklearn.metrics import mean_absolute_percentage_error as mape
# mse
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ticker  = 'AAPL'
pre_days = 1

number_of_training = 2300

filePath = './model/LSTM/final_model'
# model_dir = './model/LSTM/final_model/AAPL_with_trend_1.57_17_window_length_1_lstm_layers_2_dense_1_unit_170'
# model_dir = './model/LSTM/final_model/MSFT_without_trend_1.51_24_window_length_1_lstm_layers_1_dense_1_unit_160'
all_result_list = os.listdir(filePath)

for part_of_dir in all_result_list:
    model_dir = filePath + '/' + part_of_dir
    print(model_dir)
    ticker = model_dir.split(sep='./model/LSTM/final_model/')[1].split(sep='_')[0]
    flag = model_dir.split(sep='_')[2]
    if flag == 'with':
        features = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume', ticker]
    else:
        features = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']

    # using memory_his_days to predict.
    memory_his_days = eval(model_dir.split(sep='_window_length_')[1].split(sep='_')[0])


    def stock_price_LSTM_pre_processing(df, memory_his_days, pre_days=pre_days):
        pd.set_option('display.max_columns', None)
        df.dropna(inplace=True)
        df['Y'] = df['Close'].shift(-pre_days)
        # standardize
        scalar_X = StandardScaler()
        # scalar_Y = MinMaxScaler()

        sca_X = scalar_X.fit_transform(df[features].values)

        # construct feature spaces
        X = []
        deq = deque(maxlen=memory_his_days)
        for data in sca_X:
            deq.append(list(data))
            if len(deq) == memory_his_days:
                X.append(list(deq))
        # last data is not enough to construct the training set

        # discard tail due to lack of Y, preserve discarded data in variable x_last
        X_last = sca_X[-pre_days:, ]
        X = np.array(X[:-pre_days])
        # construct Y
        Y = df['Y'][memory_his_days - 1:-pre_days]
        number_of_instance = len(Y)
        Y = np.array(Y)
        # train test split
        X_train = X[:number_of_training, :, :]
        X_test = X[number_of_training:, :, :]
        Y_train = Y[:number_of_training]
        Y_test = Y[number_of_training:]
        return scalar_X, X_train, X_test, Y_train, Y_test


    scalar_X, X_train, X_test, Y_train, Y_test = stock_price_LSTM_pre_processing(
        pd.read_csv('./Feature_Engineering/result/' + ticker + '.csv').drop('Unnamed: 0', axis=1), memory_his_days)

    # saving architecture
    model = load_model(model_dir)
    model.summary()
    # src = model_to_dot(model,show_shapes=True,
    #      show_layer_names=True,
    #      rankdir='TB',
    #      dpi=300, expand_nested=True,subgraph = False)
    # # src.write_png("AAPL.png")
    # src.write_png("AAPL_with_trend.png")

    # prediction
    pred = model.predict(X_test)
    predict = [np.average(x) for x in pred]
    font_size_big = 18
    font_size_small = 17

    fig = plt.figure(figsize=(7, 7))
    plt.plot(Y_test)
    plt.plot(predict)
    plt.ylabel('Price', fontsize=font_size_big)
    plt.yticks(fontsize=font_size_big)
    plt.xticks(range(0, 250, 50), fontsize=font_size_big)  # 个数 这里是三个是一个间隔  共10个  和下面的一样
    plt.legend(['true', 'predicted'], fontsize=font_size_small)
    plt.xticks(range(0, 250, 50), fontsize=font_size_big)  # 个数 这里是三个是一个间隔  共10个  和下面的一样
    plt.xlabel('No of days', fontsize=font_size_big)

    if flag == 'with':
        title = 'LSTM prediction of ' + ticker + ' with google trend index'
    else:
        title = 'LSTM prediction of ' + ticker + ' without google trend index'
    plt.title(title, fontsize=font_size_big)
    plt.grid()
    # if flag =='without':
    #     plt.savefig(f'LSTM_{ticker}_without_trend_line.png')
    # else:
    #     plt.savefig(f'LSTM_{ticker}_with_trend_line.png')
    plt.show()

    if flag == 'without':
        print(f'LSTM_{ticker}_without_trend')
    else:
        print(f'LSTM_{ticker}_with_trend')
    print('rmse:', mse(Y_test, predict, squared=False))
    print('mae:', mae(Y_test, predict))
    print('mape:', mape(Y_test, predict))