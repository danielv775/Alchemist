from os import close
import pandas as pd
from pandas.core.frame import DataFrame
from datetime import datetime
import datetime as dt

from alchemist.src.etl.data_loader import load_market_data
from alchemist.src.consts import *

def create_initial_portfolio(initial_cash: float, symbols: list[str], initial_holdings: list[float], start_date: datetime) -> DataFrame:
    """Creates an initial portfolio dataframe

    Args:
        initial_cash (float): Initial cash available for investing
        symbols (list[str]): Symbols of initial stocks/cryptos
        initial_holdings (list[float]): Quantities of initial stocks/cryptos
        start_date (datetime): Starting date of the portfolio

    Returns:
        DataFrame: Portfolio dataframe like below:

                           CASH  NOM   TSLA   SPY  PortValue
            2019-01-01  10000.0  0.0  100.0  30.0  NaN
            
    """   
    closest_day = get_closest_future_market_day(start_date)

    if closest_day != start_date:
        print(f'Warning... start_date {start_date} is not a market day. Using the closest future day instead {closest_day}')
        start_date = closest_day

    portfolio = pd.DataFrame(columns=[CASH , *symbols, PORT_VALUE], index=[start_date])

    portfolio.loc[start_date, CASH] = initial_cash

    portfolio.loc[start_date, symbols] = initial_holdings

    portfolio = portfolio.astype('float64')

    return portfolio


def get_closest_future_market_day(date: datetime):

    # Loading daily market data
    market_data = load_market_data(
                ['SPY'],
                date,
                date + dt.timedelta(days=14),
                return_dict=False,
                invalidate_cache=False,
            )    
    
    result = market_data.index[0]

    return result
