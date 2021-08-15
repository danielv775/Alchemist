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
    return_dict:bool=True
):
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
        return_dict(bool, optional): True to return a dict of DataFrames or False to return a multiindex DataFrame

    Returns:
        [dict]: Dictionary of dataframes {'stock_symbol': pd.DataFrame}
        or
        [DataFrame]: Multiindex dataframes with axes[0] Date Index and axes[1] a Multiindex of [Symbol, Features]
    """

    market_data_dict = {}
    
    if not invalidate_cache:
        
        try:
            market_data_dict = read_cached_market_data(symbols, cache_folder, intraday)            
        
        except FileNotFoundError:
            invalidate_cache = True


    if invalidate_cache:

        if intraday:                    
                
            # Get brand new data
            market_data_dict = download_intraday_market_data(
                symbols=symbols,
                months=months,
                interval=interval,
                cache_folder=cache_folder
            ) 

        else: 

            market_data_dict = download_daily_market_data(
                symbols,
                start_date,
                end_date,
                cache_folder=cache_folder
            )

    # Returns either a dict of DataFrames or a multiindex DataFrame
    if return_dict:
        # Returns a dict of DataFrames where keys are the stock symbols, for example:
        # {'TSLA':                   Hi...6 columns], 'AAPL':                   Hi...6 columns]}

        return market_data_dict
    
    else:
        # Returns a Multiindex DataFrame, for example:
        #                   TSLA                                                                    AAPL                                                             
        #                   High         Low        Open       Close      Volume    AdjClose        High         Low        Open       Close       Volume    AdjClose
        # Date                                                                                                                                                       
        # 2019-01-02   63.026001   59.759998   61.220001   62.023998  58293000.0   62.023998   39.712502   38.557499   38.722500   39.480000  148158800.0   38.382229
        # 2019-01-03   61.880001   59.476002   61.400002   60.071999  34826000.0   60.071999   36.430000   35.500000   35.994999   35.547501  365248800.0   34.559078
        # 2019-01-04   63.599998   60.546001   61.200001   63.537998  36970500.0   63.537998   37.137501   35.950001   36.132500   37.064999  234428400.0   36.034370        
        
        market_data_df = pd.concat(market_data_dict, axis=1)

        market_data_df = market_data_df.rename_axis(DATE).rename_axis([SYMBOLS, FEATURES], axis='columns')

        return market_data_df


def read_cached_market_data(symbols:list, cache_folder:str, intraday:bool) -> dict:    

    market_data = {}

    get_filename_func = _get_intraday_filename if intraday else _get_daily_filename

    for symbol in symbols:
        
        filename = get_filename_func(cache_folder, symbol)

        try:
            data = pd.read_csv(filename, parse_dates=[DATE])

            data.set_index(DATE, inplace=True)
            
            # Add name to columns, for example:
            # From this:
            #                   High         Low        Open       Close      Volume    AdjClose
            # Date                                                                              
            # 2019-01-02   63.026001   59.759998   61.220001   62.023998  58293000.0   62.023998
            # 2019-01-03   61.880001   59.476002   61.400002   60.071999  34826000.0   60.071999
            #
            # To this:
            # Features          High         Low        Open       Close      Volume    AdjClose
            # Date                                                                              
            # 2019-01-02   63.026001   59.759998   61.220001   62.023998  58293000.0   62.023998
            # 2019-01-03   61.880001   59.476002   61.400002   60.071999  34826000.0   60.071999
            data = data.rename_axis(DATE).rename_axis(FEATURES, axis='columns')

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

        # Renaming column names
        df = df.rename(
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

        # Add name to columns, for example:
        # From this:
        #                   High         Low        Open       Close      Volume    AdjClose
        # Date                                                                              
        # 2019-01-02   63.026001   59.759998   61.220001   62.023998  58293000.0   62.023998
        # 2019-01-03   61.880001   59.476002   61.400002   60.071999  34826000.0   60.071999
        #
        # To this:
        # Features          High         Low        Open       Close      Volume    AdjClose
        # Date                                                                              
        # 2019-01-02   63.026001   59.759998   61.220001   62.023998  58293000.0   62.023998
        # 2019-01-03   61.880001   59.476002   61.400002   60.071999  34826000.0   60.071999
        df = df.rename_axis(DATE).rename_axis(FEATURES, axis='columns')

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

        # Add name to columns, for example:
        # From this:
        #                   High         Low        Open       Close      Volume    AdjClose
        # Date                                                                              
        # 2019-01-02   63.026001   59.759998   61.220001   62.023998  58293000.0   62.023998
        # 2019-01-03   61.880001   59.476002   61.400002   60.071999  34826000.0   60.071999
        #
        # To this:
        # Features          High         Low        Open       Close      Volume    AdjClose
        # Date                                                                              
        # 2019-01-02   63.026001   59.759998   61.220001   62.023998  58293000.0   62.023998
        # 2019-01-03   61.880001   59.476002   61.400002   60.071999  34826000.0   60.071999
        data = data.rename_axis(DATE).rename_axis(FEATURES, axis='columns')            

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
