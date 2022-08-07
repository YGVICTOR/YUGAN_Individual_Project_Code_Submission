"""
    This script gives a descriptive analysis for the dataset.
    Specifically, the script will do the following  tings:
    1. Plot the daily closing price of AAPL, AMZN, GOOG, META, and MSFT.
    2. Compute the mean, standard deviation and median of the closing price for AAPL, AMZN, GOOG, META and MSFT.
    3. Conduct the Jarque-Bera test for daily closing price for AAPL, AMZN, GOOG, META, and MSFT.
    4. Apply the log transform of daily closing price for AAPL, AMZN, GOOG, META, and MSFT and Plot them.
    5. Plot the histogram of daily log returns for AAPL, AMZN, GOOG, MSFT, and META.
    6. Compute the skew and kurt of daily log returns for AAPL, AMZN, GOOG, MSFT, and META.
    7. Plot the QQ plot of daily log return for AAPL, AMZN, GOOG, MSFT, and META.
    8. Plot the auto-correlation of daily log return for AAPL, AMZN, GOOG, MSFT, and META.
    9. Conduct the ADF test for daily log return for AAPL, AMZN, GOOG, MSFT, and META.
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics import tsaplots

stocks = ['AAPL','AMZN','FB','GOOG','MSFT']

font_size_big = 20
font_size_small = 20
plt.figure(figsize=(20,20))
for index,stock in enumerate(stocks):
    if index +1 >4:
        break
    ax = plt.subplot(2,2,index+1)
    file_path = '../result/raw_data/'+stock+'.csv'
    data_frame = pd.read_csv(file_path, parse_dates=True).drop('Unnamed: 0', axis=1)
    plt.plot(data_frame['Date'],data_frame['Close'], color='r')

    plt.xticks(range(0, 2519, 250),rotation=40)  # 个数 这里是三个是一个间隔  共10个  和下面的一样
    plt.yticks( fontsize=font_size_big)
    ax.set_xticklabels(data_frame.apply(lambda x: x['Date'][:4],axis=1)[::250],fontsize=font_size_small)
    ax.set_xlabel('Date',fontsize=font_size_small)
    plt.ylabel('Price',fontsize=font_size_big)
    title = stock + ' price from Jun 18th 2012 to Jun 17th 2022'
    plt.title(title,fontsize=font_size_big)
    plt.grid()
# plt.savefig('close_price_4.png')
plt.show()

plt.figure(figsize=(12,9))
ax = plt.subplot(1,1,1)
file_path = '../result/raw_data/'+stocks[-1]+'.csv'
data_frame = pd.read_csv(file_path, parse_dates=True).drop('Unnamed: 0', axis=1)
plt.plot(data_frame['Date'], data_frame['Close'], color='r')
plt.xticks(range(0, 2519, 250),rotation=40)  # 个数 这里是三个是一个间隔  共10个  和下面的一样
plt.yticks( fontsize=font_size_big)
ax.set_xticklabels(data_frame.apply(lambda x: x['Date'][:4],axis=1)[::250],fontsize=font_size_small)
ax.set_xlabel('Date',fontsize=font_size_small)
plt.ylabel('Price',fontsize=font_size_big)
title = stocks[-1] + ' price from Jun 18th 2012 to Jun 17th 2022'
plt.title(title,fontsize=font_size_big)
plt.grid()
# plt.savefig('close_price_1.png')
plt.show()

# ADF test
from statsmodels.tsa.stattools import adfuller
for stock in stocks:
    file_path = '../result/raw_data/' + stock + '.csv'
    data_frame = pd.read_csv(file_path, parse_dates=True).drop('Unnamed: 0', axis=1)
    adf = adfuller(data_frame['Close'], maxlag=12)
    print("######################### ADF test for "+stock+" #########################")
    print("\nStatistics analysis\n")
    print("Statistic Test : ", adf[0])
    print("p-value : ", adf[1])
    print("# n_lags : ", adf[2])
    print("No of observation: ", adf[3])
    for key,value in adf[4].items():
        print(f" critical value {key} : {value}")


    print("######################### end of ADF Test #########################")


print('begin')
# apply the log first order difference
for index,stock in enumerate(stocks):
    file_path = '../result/raw_data/' + stock + '.csv'
    data_frame = pd.read_csv(file_path, parse_dates=True).drop('Unnamed: 0', axis=1)
    data_frame['log_return'] = np.log(data_frame.Close) - np.log(data_frame.Close.shift(1))

    fig = plt.figure(figsize=(12, 9.7))
    ax = plt.subplot(1, 1, 1)
    # file_path = 'data/result/raw_data/' + stocks[-1] + '.csv'
    # data_frame = pd.read_csv(file_path, parse_dates=True).drop('Unnamed: 0', axis=1)
    plt.plot(data_frame['Date'],data_frame['log_return'], color='r')
    plt.xticks(range(0, 2519, 250), rotation=40)  # 个数 这里是三个是一个间隔  共10个  和下面的一样
    plt.yticks(fontsize=font_size_big)
    ax.set_xticklabels(data_frame.apply(lambda x: x['Date'][:4], axis=1)[::250], fontsize=font_size_small)
    ax.set_xlabel('Date', fontsize=font_size_small)
    plt.ylabel('Daily Return', fontsize=font_size_big)
    title = stock + ' daily long returns from Jun 19th 2012 to Jun 17th 2022'
    plt.title(title, fontsize=font_size_big)
    plt.grid()
    save = stock+"_daily_log_return.png"
    # plt.savefig(save)
    plt.show()


for index,stock in enumerate(stocks):
    file_path = '../result/raw_data/' + stock + '.csv'
    data_frame = pd.read_csv(file_path, parse_dates=True).drop('Unnamed: 0', axis=1)
    data_frame['log_return'] = np.log(data_frame.Close) - np.log(data_frame.Close.shift(1))
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(1, 1, 1)
    # file_path = 'data/result/raw_data/' + stocks[-1] + '.csv'
    # data_frame = pd.read_csv(file_path, parse_dates=True).drop('Unnamed: 0', axis=1)
    plt.hist(data_frame['log_return'],bins=50, color='r')
    plt.yticks(fontsize=font_size_big)
    plt.tick_params(labelsize=font_size_small)
    plt.grid()
    title = "Histogram of " + stock + ' daily long returns'
    plt.title(title, fontsize=font_size_big)
    save = "histogram_"+ stock + "_daily_log_return.png"
    # plt.savefig(save)
    plt.show()

import pylab
import scipy.stats as stats

# pair plot
log_df = pd.DataFrame()
for index,stock in enumerate(stocks):
    file_path = '../result/raw_data/' + stock + '.csv'
    data_frame = pd.read_csv(file_path, parse_dates=True).drop('Unnamed: 0', axis=1)
    data_frame['log_return'] = np.log(data_frame.Close) - np.log(data_frame.Close.shift(1))
    # deal with Kurtosis and Skewness
    log_df[stock] = data_frame['log_return']
    print(stock)
    print('skew',data_frame['log_return'].skew())
    print('kurt',data_frame['log_return'].kurt())
    print()

    # qq plot
    # fig = plt.figure()
    # stats.probplot(data_frame['log_return'], dist="norm", plot=plt)
    # title = 'Normal QQ plot of '+ stock +' daily returns'
    # plt.title(title)
    # plt.show()


    fig = plt.figure(figsize=(6.4,6.3))
    measurements = data_frame['log_return']
    stats.probplot(measurements, plot=pylab)
    title = 'QQ plot of '+ stock
    plt.title(title,fontsize=font_size_big)
    plt.tick_params(labelsize=font_size_small)
    plt.xlabel('Theoretical quantiles',fontsize=font_size_small)
    save = 'qqPlot_'+stock+'.png'
    # plt.savefig(save)
    pylab.show()


    data_frame = data_frame.dropna()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(visible=True)
    tsaplots.plot_acf(data_frame['log_return'], lags=2500, ax=ax)
    ax.set_ylim(-0.1, 0.1)
    title = 'autocorrelation of '+stock+' daily log return'
    ax.set_title(title)
    ax.set_ylabel('autocorrelation')
    save = 'autocorrelation_' +stock+'.png'
    # plt.savefig(save)
    plt.show()


result_dict ={
    'AAPL':[],
    'AMZN':[],
    'GOOG':[],
    'MSFT':[],
    'FB':[]
}
for index,stock in enumerate(stocks):
    file_path = '../result/raw_data/'+stock+'.csv'
    data_frame = pd.read_csv(file_path, parse_dates=True).drop('Unnamed: 0', axis=1)
    result_dict[stock] = data_frame['Close'].tolist()

stock_Close_df = pd.DataFrame(result_dict)
from scipy.stats import jarque_bera

for index, stock in enumerate(stocks):
    res =  (jarque_bera(stock_Close_df[stock]))
    print(stock)
    print(res)
    print(stock,'mean:',np.mean(stock_Close_df[stock]),'Std. Dev.:',np.std(stock_Close_df[stock]),'median:',np.median(stock_Close_df[stock]),'Observation',len(stock_Close_df[stock]),'Jarque-Bera',res[0],'p-value:',res[1])
