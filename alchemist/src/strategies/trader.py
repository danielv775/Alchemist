
from abc import ABC, abstractmethod, abstractproperty

from datetime import datetime
import datetime as dt 

from pandas.core.frame import DataFrame

class Trader(ABC):

    def __init__(self, name: str, impact: float=0.0):
        """Init for Abstract Trader 

        Args:
            name (str): Name to describe trader (i.e. Rules, Basic, Model, etc.)
        """
        self.name = name
        self.impact = impact
    
    @abstractmethod
    def trade(self, symbol: str, start_date: datetime, end_date: datetime, start_value: float) -> DataFrame:
        
        """Trade with trained model or with a rules based strategy

        Args:
            symbol (str): asset ticker (i.e. TSLA, SQ)
            start_date (datetime): start date
            end_date (datetime): end date
            start_value (float): initial USD

        Returns:
            DataFrame: DataFrame containing trades made on each day, defined by change in any asset +/-
        """
        pass
    
    def __str__(self):
        return self.name
        
if __name__ == '__main__':
    pass