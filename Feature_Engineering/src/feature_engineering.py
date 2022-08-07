'''
    Using this script to concatenate the technical indicators and the google trends.
'''

import pandas as pd
import datetime
stocks = ['AAPL','AMZN','FB','GOOG','MSFT']


for stock in stocks:
    trend_location = '../../data/result/google_trends/'+stock+'_2012-01-01_2022-10-01.csv'
    row_location = '../../data/result/raw_data/'+stock+'.csv'
    trend_df = pd.read_csv(trend_location,parse_dates=True)
    raw_df = pd.read_csv(row_location, parse_dates=True).drop('Unnamed: 0', axis=1)
    fmt = '%Y/%m/%d'
    trend_df['Date'] = trend_df.apply(lambda x: datetime.datetime.strptime(x['Date'],fmt).strftime('%Y-%m-%d'),axis=1)
    target_df = pd.merge(raw_df,trend_df,on='Date',how='left')
    file_path = '../result/'+stock+'.csv'
    print(target_df)
    # target_df.to_csv(file_path)