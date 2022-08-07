"""
    Using this script to download the google trends and apply the overlapping window
    transforming.
"""

import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm,trange

def trendsDaysThreeMonthApi(kw_list,start_date,end_date):
    pytrend = TrendReq(hl='en-US',tz=360)
    pytrend.build_payload(kw_list=kw_list,timeframe=start_date+' '+end_date)
    interest_over_time_df = pytrend.interest_over_time()
    interest_over_time_df = interest_over_time_df[kw_list]
    print(interest_over_time_df)
    '''
        uncomment code below to save the file.
    '''
    # interest_over_time_df.to_csv(f'../google_trend_detail/x/{kw_list[0]}_{start_date}_{end_date}.csv')


def trendsDaysThreeMonthApiOptionBaseIndex(kw_list,start_date,end_date,is_base_index,base_index):
    pytrend = TrendReq(hl='en-US',tz=360)
    pytrend.build_payload(kw_list=kw_list,timeframe=start_date+' '+end_date)
    interest_over_time_df = pytrend.interest_over_time()
    interest_over_time_df = interest_over_time_df[kw_list]

    if is_base_index != 0:
        interest_over_time_df = interest_over_time_df * (base_index/interest_over_time_df.iloc[-1,:])
    return interest_over_time_df


# parameters
kw_list = [['AMZN'],['AAPL'],['META'],['GOOG'],['MSFT']]
start_year = 2012
end_year = 2022

# split data into 3 months timewindow
years = [str(i) for i in range(start_year,end_year+1)]
months = ['01','04','07','10']
days_break = [year+'-'+month+'-'+'01' for year in years for month in months]
days_break_tuple = [(days_break[i],days_break[i+1]) for i in range(len(days_break)-1)]

# save data file
for kw in kw_list:
    for day_break_tumple in tqdm(days_break_tuple):
        trendsDaysThreeMonthApi(kw,day_break_tumple[0],day_break_tumple[1])

# convert
for kw in kw_list:
    # initiate
    days_tuple = days_break_tuple[-1]
    result = trendsDaysThreeMonthApiOptionBaseIndex(kw,days_tuple[0],days_tuple[1],0,0)

    for days_tuple in tqdm(days_break_tuple[-2::-1]):
        days_tuple_df = trendsDaysThreeMonthApiOptionBaseIndex(kw,days_tuple[0],days_tuple[1],1,result.iloc[0,])
        result = pd.concat([days_tuple_df.iloc[:-1,],result])
        '''
            uncomment code below to save the file.
        '''
    # result.to_csv(f'../result/google_trends/{kw[0]}_{days_break_tuple[0][0]}_{days_break_tuple[-1][-1]}.csv')
    print(result)