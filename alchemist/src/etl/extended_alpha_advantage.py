import csv
import requests
import time
import pandas as pd
import os
from datetime import date, datetime

from alchemist.src.consts import *


def get_intraday_data(symbol:str, api_key:str, months:int, interval:str='60min') -> pd.DataFrame:
    """Get intraday data using Alpha Advantage API

    Args:
        symbol (str): Stock symbol
        api_key (str): Alpha advantage API key
        months (int): number of last months to get data from
        interval (str, optional): Intraday interval

    Returns:
        pd.DataFrame: market data in ascending order by time (more recent on the bottom)
    """

    if months > 24:
        ValueError('Alpha Advantage intraday has a limit of 2 years')    

    slices = _get_time_slices(months)

    # Break Up into groups of 5 due to Rate Limit
    slices_split = [slices[i:i+5] for i in range(0, len(slices), 5)]

    with requests.Session() as s:

        historic_data = pd.DataFrame()

        for index, time_slices in enumerate(slices_split, start=1):

            df_slices = _get_time_series_intraday_extended_multi_slice(
                session=s,
                sym=symbol,
                interval=interval,
                time_slices=time_slices,
                api_key=api_key,
            )

            historic_data = pd.concat([historic_data, df_slices])

            print(f'Current historic data shape: {historic_data.shape}')

            # Wait for 60s due to API call limit of 5 calls per minute
            if index < len(slices_split):
                print('Made ~5 Calls <1 Min. Wait a Minute...')
                time.sleep(60)
   
    

    historic_data = historic_data.rename(
            columns={
                    'time': DATE,
                    'high': HIGH,
                     'low': LOW,
                     'open': OPEN,
                     'close': CLOSE,
                     'volume': VOLUME,
                    #  'Adj Close': ADJUSTED_CLOSE,  # Adjusted Data is already download by default
                     }
        )

    historic_data = historic_data.set_index(DATE)        

    # reverse since alphavantage returns descending by date
    historic_data = historic_data[::-1]

    # Renaming columns


    return historic_data


def _get_time_slices(num_months):
    """Returns a list of strings of data slices. 
    For example, if num_months = 3, it returs ['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6']

    Args:
        num_months (int): Number of previous months

    Returns:
        list: List of yearXmonthY
    """

    slices = []

    for month in range(1, num_months + 1):

        year = (month - 1) // 12 + 1

        month = month - 12 * (year-1)

        slices.append(f'year{year}month{month}')

    return slices


def _get_time_series_intraday_extended(session, sym, interval, time_slice, api_key):

    # https://www.alphavantage.co/documentation/#intraday-extended

    print(f'Pulling intraday time series data for time_slice: {time_slice}')

    URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={sym}&interval={interval}&slice={time_slice}&adjusted=true&apikey={api_key}'

    download = session.get(URL)
    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    data_list = list(cr)

    df_slice = pd.DataFrame(data_list)
    df_slice.columns = df_slice.iloc[0]
    df_slice = df_slice.iloc[1:, :]

    return df_slice


def _get_time_series_intraday_extended_multi_slice(session, sym, interval, time_slices, api_key):

    df_slices = pd.DataFrame()

    for time_slice in time_slices:

        df_slice = _get_time_series_intraday_extended(
            session=session,
            sym=sym,
            interval=interval,
            time_slice=time_slice,
            api_key=api_key,
        )

        df_slices = pd.concat([df_slices, df_slice])

    return df_slices


if __name__ == '__main__':
    a = _get_time_slices(6)

    print(a)