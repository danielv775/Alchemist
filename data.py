import csv
import requests
import time
import pandas as pd
import os

ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA_VANTAGE_TOKEN')

def get_time_series_intraday_extended(session, sym, interval, time_slice):

    # https://www.alphavantage.co/documentation/#intraday-extended
    
    print(f'Pulling intraday time series data for time_slice: {time_slice}')

    URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={sym}&interval={interval}&slice={time_slice}&adjusted=true&apikey={ALPHA_VANTAGE_TOKEN}'

    download = session.get(URL)
    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    data_list = list(cr)

    df_slice = pd.DataFrame(data_list)
    df_slice.columns = df_slice.iloc[0]
    df_slice = df_slice.iloc[1:, :]
    
    return df_slice

def get_time_series_intraday_extended_multi_slice(session, sym, interval, time_slices):

    df_slices = pd.DataFrame()

    for time_slice in time_slices:
        
        df_slice = get_time_series_intraday_extended(session=session,
                                        sym=sym, 
                                        interval=interval, 
                                        time_slice=time_slice)

        df_slices = pd.concat([df_slices, df_slice])

    return df_slices

if __name__ == '__main__':
    pass
