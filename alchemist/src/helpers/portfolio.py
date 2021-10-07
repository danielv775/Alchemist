import pandas as pd
from pandas.core.frame import DataFrame
from datetime import datetime

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

                           CASH  NOM   TSLA   SPY
            2019-01-01  10000.0  0.0  100.0  30.0
            
    """   

    portfolio = pd.DataFrame(columns=[CASH , *symbols], index=[start_date])

    portfolio.loc[start_date, CASH] = initial_cash

    portfolio.loc[start_date, symbols] = initial_holdings

    portfolio = portfolio.astype('float64')

    return portfolio



    

