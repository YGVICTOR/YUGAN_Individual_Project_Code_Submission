'''
This script plot the results of the project. Specifically, This script will do the following 4 things.
1. Plot the forecasted stock prices for original, LSTM, TCN, GRU of
  AAPL, AMZN, MSFT, META,  and GOOG with google trends.

2. Plot the forecasted stock prices for original, LSTM, TCN, GRU of
  AAPL, AMZN, MSFT, META, and GOOG without google trends.

3. Plot the comparison of average trainable parameters for 3 methods.

4. Plot the comparison of MAPE with and without google trends as predictors
   for AAPL, AMZN, MSFT, META, and GOOG.

'''

import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import model_to_dot
import os
from matplotlib import pyplot as plt

pre_days = 1
number_of_training = 2300
result = {
    'AAPL': [],
    'AMZN': [],
    'GOOG': [],
    'MSFT': [],
    'META': []
}


def stock_price_pre_processing(df, memory_his_days, features_, pre_days=pre_days):
    pd.set_option('display.max_columns', None)
    df.dropna(inplace=True)
    df['Y'] = df['Close'].shift(-pre_days)
    # standardize
    scalar_X = StandardScaler()
    # scalar_Y = MinMaxScaler()

    sca_X = scalar_X.fit_transform(df[features_].values)

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


model_name_list = ['TCN', 'GRU', 'LSTM']
tick_list = ['AAPL', 'AMZN', 'MSFT', 'META', 'GOOG']

# 1. The graphs show forecasted stock prices for original, Long Short-Term Memory,
# Gated Recurrent Unit, and Temporal Convolutional Neural Network of AAPL,
# AMZN, GOOG, META, and MSFT using google_trends.
for model_name in model_name_list:
    filePath = './model/' + model_name + '/final_model'
    all_result_list = os.listdir(filePath)
    for part_of_dir in all_result_list:
        model_dir = filePath + '/' + part_of_dir
        sep = filePath + '/'
        ticker = model_dir.split(sep=sep)[1].split(sep='_')[0]
        features = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume', ticker]
        flag = model_dir.split(sep='_')[2]
        window_length = eval(model_dir.split(sep='_window_length_')[1].split(sep='_')[0])
        if flag == 'with':
            print(part_of_dir)
            scalar_X, X_train, X_test, Y_train, Y_test = stock_price_pre_processing(
                pd.read_csv('./Feature_Engineering/result/' + ticker + '.csv').drop('Unnamed: 0', axis=1),
                memory_his_days=window_length, features_=features)
            model = load_model(model_dir)
            pred = model.predict(X_test)
            predict = [np.average(x) for x in pred]
            result[ticker].append((model_name, predict))
for ticker in result.keys():
    scalar_X, X_train, X_test, Y_train, Y_test = stock_price_pre_processing(
        pd.read_csv('./Feature_Engineering/result/' + ticker + '.csv').drop('Unnamed: 0', axis=1),
        memory_his_days=window_length, features_=['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'])
    result[ticker].append(('Original', Y_test))
for key, item in result.items():
    font_size_big = 18
    font_size_small = 17
    fig = plt.figure(figsize=(8.5, 7))
    legend = []
    for model_value_set in item:
        legend.append(model_value_set[0])
        plt.plot(model_value_set[1])
    plt.ylabel('Price', fontsize=font_size_big)
    plt.yticks(fontsize=font_size_big)
    plt.xticks(range(0, 250, 50), fontsize=font_size_big)  # 个数 这里是三个是一个间隔  共10个  和下面的一样
    plt.xlabel('No of days', fontsize=font_size_big)
    plt.legend(legend, fontsize=font_size_big)
    title = key + ' with google trends'
    plt.title(title, fontsize=font_size_big)
    plt.grid()
    save_path = key + '_compare_with_trend.png'
    # plt.savefig(save_path)
    plt.show()

# 2 .The graphs show forecasted stock prices for original, Long Short-Term Memory,
# Gated Recurrent Unit, and Temporal Convolutional Neural Network of AAPL,
# AMZN, GOOG, META, and MSFT without using google_trends.
result = {
    'AAPL': [],
    'AMZN': [],
    'GOOG': [],
    'MSFT': [],
    'META': []
}
for model_name in model_name_list:
    filePath = './model/' + model_name + '/final_model'
    all_result_list = os.listdir(filePath)
    for part_of_dir in all_result_list:
        model_dir = filePath + '/' + part_of_dir
        sep = filePath + '/'
        ticker = model_dir.split(sep=sep)[1].split(sep='_')[0]
        features = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
        flag = model_dir.split(sep='_')[2]
        window_length = eval(model_dir.split(sep='_window_length_')[1].split(sep='_')[0])
        if flag == 'without':
            print(part_of_dir)
            scalar_X, X_train, X_test, Y_train, Y_test = stock_price_pre_processing(
                pd.read_csv('./Feature_Engineering/result/' + ticker + '.csv').drop('Unnamed: 0', axis=1),
                memory_his_days=window_length, features_=features)
            model = load_model(model_dir)
            pred = model.predict(X_test)
            predict = [np.average(x) for x in pred]
            result[ticker].append((model_name, predict))
for ticker in result.keys():
    scalar_X, X_train, X_test, Y_train, Y_test = stock_price_pre_processing(
        pd.read_csv('./Feature_Engineering/result/' + ticker + '.csv').drop('Unnamed: 0', axis=1),
        memory_his_days=window_length, features_=['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'])
    result[ticker].append(('Original', Y_test))
for key, item in result.items():
    font_size_big = 18
    font_size_small = 17
    fig = plt.figure(figsize=(8.5, 7))
    legend = []
    for model_value_set in item:
        legend.append(model_value_set[0])
        plt.plot(model_value_set[1])
    plt.ylabel('Price', fontsize=font_size_big)
    plt.yticks(fontsize=font_size_big)
    plt.xticks(range(0, 250, 50), fontsize=font_size_big)  # 个数 这里是三个是一个间隔  共10个  和下面的一样
    plt.xlabel('No of days', fontsize=font_size_big)
    plt.legend(legend, fontsize=font_size_big)
    title = key + ' without google trends'
    plt.title(title, fontsize=font_size_big)
    plt.grid()
    save_path = key + '_compare_with_trend.png'
    # plt.savefig(save_path)
    plt.show()

# 3. The average trainable parameters for 3 methods.
# construct data
font_size_big = 18
font_size_small = 15
remove_factor = 0
x_data = ['GRU', 'TCN', 'LSTM']
ticker_list = ['AAPL', 'AMZN', 'MSFT', 'GOOG', 'META']
plt.figure(figsize=(8, 6))
plt.rc('font', size=font_size_big)
ax = plt.gca()  # gca:get current axis
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
total_params = [343259, 561444, 718180]
bar_width = 0.3
plt.xticks(fontsize=font_size_small)
plt.barh(remove_factor + np.arange(len(x_data)), total_params, height=bar_width, color='orange')
# ha horizontal, va vertical
for x, y in enumerate(total_params):
    plt.text(y - 18000, x, '%s' % y, ha='right', va='center', fontsize=font_size_big)
# rest axis
plt.xlim(0, 700000)
plt.xticks(np.arange(0, 800000, 100000))
plt.xticks(fontsize=font_size_big)
plt.yticks(np.arange(len(x_data)), x_data, fontsize=font_size_big)
plt.xlabel('the number of trainable parameters', fontsize=font_size_big)
# plt.legend(["without google trends", "with google trends"], loc=2, fontsize=font_size_small)
title_content = 'the average trainable parameters for 3 methods'
plt.title(title_content, fontsize=font_size_big)
save_file = 'trainable_param.png'
# plt.savefig(save_file)
plt.show()

## 4. The average storing spaces needed for 3 methods.
plt.figure(figsize=(8, 6))
plt.rc('font', size=font_size_big)
ax = plt.gca()  # gca:get current
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
total_params = [4.659, 7.484, 9]
bar_width = 0.3
plt.xticks(fontsize=font_size_small)
plt.barh(remove_factor + np.arange(len(x_data)), total_params, height=bar_width, color='orange')
# ha horizontal, va vertical
for x, y in enumerate(total_params):
    plt.text(y, x, '%s' % y, ha='right', va='center', fontsize=font_size_big)
# rest axis
plt.xlim(0, 10)
plt.xticks(np.arange(0, 10, 1))
plt.xticks(fontsize=font_size_big)
plt.yticks(np.arange(len(x_data)), x_data, fontsize=font_size_big)
plt.xlabel('storing space (MB)', fontsize=font_size_big)
# plt.legend(["without google trends", "with google trends"], loc=2, fontsize=font_size_small)
title_content = 'the average storing spaces needed for 3 methods'
plt.title(title_content, fontsize=font_size_big)
save_file = 'storing_space.png'
# plt.savefig(save_file)
plt.show()

# 5.The graph shows the comparison of the MAPE for Long Short-Term Memory
# (LSTM), Gated Recurrent Unit (GRU), and Temporal Convolutional Neural Network
# (TCN) of AAPL, AMZN, MSFT, and GOOG
font_size_small = 15
x_data = ['LSTM', 'GRU', 'TCN']
ticker_list = ['AAPL', 'AMZN', 'MSFT', 'GOOG', 'META']
sorce = {
    'AAPL': {'without_google_trends': [0.0162, 0.0164, 0.0165],
             'with_google_trends': [0.0157, 0.0161, 0.0161]
             },
    'AMZN': {
        'without_google_trends': [0.0197, 0.0205, 0.0194],
        'with_google_trends': [0.0196, 0.0202, 0.0193],
    },
    'MSFT': {
        'without_google_trends': [0.0160, 0.0156, 0.0166],
        'with_google_trends': [0.0153, 0.0155, 0.0163]
    },
    'GOOG': {
        'without_google_trends': [0.0180, 0.0174, 0.0170],
        'with_google_trends': [0.0171, 0.0167, 0.0162]
    },
    'META': {
        'without_google_trends': [0.0235, 0.0244, 0.0226],
        'with_google_trends': [0.0267, 0.0348, 0.0442]
    }
}
for ticker in ticker_list:
    without_google_trends = sorce[ticker]['without_google_trends']
    with_google_trends = sorce[ticker]['with_google_trends']
    bar_width = 0.3
    remove_factor = 0
    max_range = max(max(np.array(without_google_trends)), max(np.array(with_google_trends)))
    print(ticker, max_range)
    plt.figure(figsize=(8, 6))
    ax = plt.gca()  # gca:get current axis
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    if max_range <= 0.0205:
        plt.xlim(0, max_range + 0.002)
        plt.xticks(np.arange(0, max_range + 0.002, 0.002), rotation=40)
    plt.xticks(fontsize=font_size_small)
    plt.barh(remove_factor + np.arange(len(x_data)), without_google_trends, height=bar_width)
    plt.barh(remove_factor + np.arange(len(x_data)) + bar_width, with_google_trends,
             height=bar_width)  # label='Java基础', color='indianred', alpha=0.8,
    # ha horizontal, va vertical
    for x, y in enumerate(without_google_trends):
        plt.text(y, remove_factor + x, '%s' % y, ha='right', va='center', fontsize=font_size_big)
    for x, y in enumerate(with_google_trends):
        plt.text(y, remove_factor + x + bar_width, '%s' % y, ha='right', va='center', fontsize=font_size_big)
    # reset axis
    plt.yticks(np.arange(len(x_data)) + bar_width / 2, x_data, fontsize=font_size_big)
    if ticker == 'META':
        plt.legend(["without google trends", "with google trends"], loc=2, fontsize=font_size_small)
    else:
        plt.legend(["without google trends", "with google trends"], loc=3, fontsize=font_size_small)
    title_content = f'The MAPE comparison of {ticker} using google trends'
    plt.title(title_content, fontsize=font_size_big)
    save_file = f'comparison_of_{ticker}_using_trends.png'
    # plt.savefig(save_file)
    plt.show()