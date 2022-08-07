"""
    Using this script to download the dataset from yfinance API.

"""
import pandas as pd
import yfinance as yf

stocks = ["AAPL", "GOOG" ,"AMZN", "FB" ,"MSFT"]
raw_data = yf.download(
        # tickers list or string as well
        tickers = stocks,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        period = "10y",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        interval = "1d",

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = False,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = False,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )

data_dict = {}
for stock in stocks:
    data_dict[stock] = raw_data[stock].copy(deep=True).reset_index()
    data_dict[stock]['Daily Return']=data_dict[stock]['Close'].pct_change()
    # store the data needed
    name = '../result/'+stock + "delete.csv"
    print(pd.DataFrame(data_dict[stock]))
    # data_dict[stock].to_csv(name)

