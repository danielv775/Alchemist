import os
import pandas as pd
import numpy as np
import datetime as dt
from copy import deepcopy
import time
from pandas_datareader import data as pdr


def download_market_data(symbols, start_date, end_date, download_folder=None):
    """ Returns a dictionary of market data for the provided symbols
    if symbols is a list of symbols, or a dataframe for a single symbol

    Args:
        symbols ([string]): Array of market symbols
        start_date (string): Start date on YYYY-MM-DD format
        end_date (string): End date on YYYY-MM-DD format
    """
    market_data = {}

    print("Getting data fromt the web")

    st = time.time()

    for symbol in symbols:

        # data = yf.download(symbol, start=start_date, end=end_date, progress=False, threads=100)
        # data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
        # share = Share(symbol)
        # data = share.get_historical(start_date, end_date)

        data = pdr.DataReader(symbol, 'yahoo-actions', start_date, end_date)

        if download_folder is not None:
            data.to_csv(f"{download_folder}/{symbol}.csv")

        market_data[symbol] = data

    print(f"Data retrieved from web in {time.time() - st} seconds")

    return market_data


if __name__ == '__main__':

    import pandas_datareader as pdr

    symbols = ['AAPL', ]
    start = dt.datetime(2021, 7, 8)   # 2021-07-08
    end = dt.datetime(2021, 8, 6)  # 2021-08-06

    ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA_VANTAGE_TOKEN')

    df = pdr.get_data_alphavantage(symbols='AAPL',
                              #function='TIME_SERIES_DAILY',
                              function='TIME_SERIES_DAILY_ADJUSTED',
                              start=start,
                              end=end,
                              retry_count=3,
                              pause=60,
                              session=None,
                              chunksize=5,
                            #   interval='60min',
                              api_key=ALPHA_VANTAGE_TOKEN)


    df.to_csv('data/AAPL2_adjusted.csv', index=True)
