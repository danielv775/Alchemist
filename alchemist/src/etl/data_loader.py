import os
import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
import time
from datetime import datetime

from pandas_datareader import data as pdr
from dotenv import load_dotenv

from alchemist.src.consts import *
from alchemist.src.etl.extended_alpha_advantage import get_intraday_data

# load .env file to os.env
load_dotenv()


def load_market_data(
    symbols: list,
    start_date: datetime = None,
    end_date: datetime = None,
    months: int = None,
    intraday: bool = False,
    interval: str = '60min',
    invalidate_cache: bool = True,
    cache_folder=f"{os.environ['PYTHONPATH']}alchemist/data",
) -> dict:
    """ Load market data

    Args:
        symbols (list): list of stock symbols
        start_date (datetime, optional): Start date used for daily market data (intraday=False)
        end_date (datetime, optional): End date used for daily market data (intraday=False)
        months (int, optional): Number of months for intraday market data (intraday=True)
        intraday (bool, optional): if set to True, it loads intraday data 
        interval (str, optional): Interval for intraday market data'60min'.
        invalidate_cache (bool, optional): if set to true, downloads market data from internet else uses csv files in cache_folder
        cache_folder (str, optional): Directory where cached market data is stored 'data'.

    Returns:
        [dict]: Dictionary of dataframes {'stock_symbol': pd.DataFrame}
    """

    market_data = {}   

    
    if not invalidate_cache:
        
        try:
            market_data = read_cached_market_data(symbols, cache_folder, intraday)            
        
        except FileNotFoundError:
            invalidate_cache = True


    if invalidate_cache:

        if intraday:                    
                
            # Get brand new data
            market_data = download_intraday_market_data(
                symbols=symbols,
                months=months,
                interval=interval,
                cache_folder=cache_folder
            ) 

        else: 

            market_data = download_daily_market_data(
                symbols,
                start_date,
                end_date,
                cache_folder=cache_folder
            )
        
    return market_data



def read_cached_market_data(symbols:list, cache_folder:str, intraday:bool) -> dict:    

    market_data = {}

    get_filename_func = _get_intraday_filename if intraday else _get_daily_filename

    for symbol in symbols:
        
        filename = get_filename_func(cache_folder, symbol)

        try:
            data = pd.read_csv(filename, parse_dates=[DATE])

            data.set_index('Date', inplace=True)
            market_data[symbol] = data

        except FileNotFoundError as e:
            # If file not exist, download data
            print('File %s does not exit', filename)
            raise e

    return market_data


def download_daily_market_data(symbols: list, start_date: datetime, end_date: datetime,
                               data_source='yahoo',
                               cache_folder=f"{os.environ['PYTHONPATH']}alchemist/data",
                               ) -> dict:
    """Download daily market data

    Args:
        symbols (list): [description]
        start_date (datetime.datetime): [description]
        end_date (datetime.datetime): [description]
        api_key ([type], optional): [description]. Defaults to None.
        data_source (str, optional): [description]. Defaults to 'yahoo-actions'.
        api_reader (str, optional): [description]. Defaults to 'pandas_datareader'.
        cache_folder (str, optional): [description]. Defaults to 'data'.

    Returns:
        [dict]: Dictionary of dataframes {'stock_symbol': pd.DataFrame}
    """
    market_data = {}

    print("Getting data fromt the web")

    api_key = None

    st = time.time()

    for symbol in symbols:

        df = pdr.DataReader(symbol, data_source, start_date, end_date)

        df.rename(
            columns={'High': HIGH,
                     'Low': LOW,
                     'Open': OPEN,
                     'Close': CLOSE,
                     'Volume': VOLUME,
                     'Adj Close': ADJUSTED_CLOSE,
                     }
        )

        if cache_folder is not None:
            df.to_csv(_get_daily_filename(cache_folder, symbol))

        market_data[symbol] = df

    print(f"Data retrieved from web in {time.time() - st} seconds")

    return market_data


def download_intraday_market_data(symbols: list,
                                  months: int,
                                  interval='60min',
                                  cache_folder=f"{os.environ['PYTHONPATH']}alchemist/data",
                                  ) -> dict:
    """Dowload intraday market data

    Args:
        symbols (list): [description]
        months (int): [description]
        interval (str, optional): [description]. Defaults to '60min'.
        cache_folder (str, optional): [description]. Defaults to 'data'.

    Returns:
        [dict]: Dictionary of dataframes {'stock_symbol': pd.DataFrame}
    """

    market_data = {}

    print("Getting data fromt the web")

    # or os.environ['ALPHA_VANTAGE_TOKEN']
    api_key = os.getenv('ALPHA_VANTAGE_TOKEN')

    st = time.time()

    for symbol in symbols:

        data = get_intraday_data(symbol, api_key, months, interval=interval)

        if cache_folder is not None:
            data.to_csv(_get_intraday_filename(cache_folder, symbol))

        market_data[symbol] = data

    print(f"Data retrieved from web in {time.time() - st} seconds")

    return market_data


def _get_intraday_filename(cache_folder, symbol):
    return f"{cache_folder}/{symbol}_INTRADAY.csv"

def _get_daily_filename(cache_folder, symbol):
    return f"{cache_folder}/{symbol}_DAILY.csv"


if __name__ == '__main__':

    start = datetime(2019, 1, 1)
    end = datetime(2020, 12, 31)

    symbols = ['TSLA', 'AAPL']

    # Loading daily market data
    daily_market_data = load_market_data(
        symbols, start, end, invalidate_cache=False)
    print(daily_market_data)

    # Loading intraday market data
    intraday_market_data = load_market_data(
        symbols, intraday=True, months=1, interval='60min', invalidate_cache=False)
    print(intraday_market_data)
