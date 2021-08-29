import pandas as pd

from alchemist.src.etl.data_loader import load_market_data
from alchemist.src.etl.technical_indicators import PriceBySMA, BBP
from alchemist.src.consts import *

from datetime import datetime
import datetime as dt 

from alchemist.src.strategies.trader import ModelTrader
from pandas.core.frame import DataFrame

class ModelTrader(Trader):

    def __init__(self, name: str, impact: float=0.0):
        super().__init__(name, impact)

    def train(self, symbol: str, start_date: datetime, end_date: datetime, start_value: float):
        """Train Model

        Args:
            symbol (str): asset ticker (i.e. TSLA, SQ)
            start_date (datetime): start date
            end_date (datetime): end date
            start_value (float): initial USD
        """
        pass
    
    def trade(self, symbol: str, start_date: datetime=(2020, 1, 1), end_date: datetime=(2020, 12, 31), start_value: float=10000) -> DataFrame:
        pass

if __name__ == '__main__':
    pass