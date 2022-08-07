'''
    Using this script to visualize the train-test split.
'''

import pandas as pd
from matplotlib import pyplot as plt

def split(dataframe, border, col):
    return dataframe.loc[:border,col], dataframe.loc[border:,col]

stockList = ['GOOG','AMZN','AAPL','META','MSFT']
# First, we get the data
df_ = {}
for i in stockList:
    file_path = '../result/'+i+".csv"
    df_[i] = pd.read_csv(file_path, index_col="Date", parse_dates=["Date"])
    df_[i] = df_[i].loc[:, ~df_[i].columns.str.contains('Unnamed')]

# train test split
df_new = {}
for i in stockList:
    df_new[i] = {}
    df_new[i]["Train"], df_new[i]["Test"] = split(df_[i], "2021-08-11", "Close")

font_size = 20
for i in stockList:
    plt.figure(figsize=(9.3,5))
    plt.plot(df_new[i]["Train"])
    plt.plot(df_new[i]["Test"])
    plt.ylabel("Price",fontsize=font_size)
    plt.yticks( fontsize=font_size)
    plt.xlabel("Date",fontsize=font_size)
    plt.xticks( fontsize=14)  # 个数 这里是三个是一个间隔  共10个  和下面的一样
    plt.legend(["Training Set", "Test Set"],fontsize=font_size)
    plt.title(i + " Closing Stock Price",fontsize=font_size)
    svg_location = i+'_train_test_split.png'
    # plt.savefig(svg_location)
    plt.show()