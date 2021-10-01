
from abc import ABC, abstractmethod
import pandas as pd
from pandas.core.frame import DataFrame

from alchemist.src.consts import *
from alchemist.src.etl.technical_indicators import ROC

class TargetProcessor(ABC):

    @abstractmethod
    def calculate_targets(self, market_data: DataFrame) -> DataFrame:
        pass


class ClassifierTargetProcessor(TargetProcessor):
    
    def __init__(self, time_ahead: int, buy_threshold: float, sell_threshold: float, impact: float=0.0) -> None:

        self.time_ahead = time_ahead
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.impact = impact


    def calculate_targets(self, market_data: DataFrame) -> DataFrame:

        # Calculating ROC and shifting by time_ahead amount
        roc = ROC(
            name='AHEAD_RETURN',
            window_size=self.time_ahead,
            price_column_name=ADJUSTED_CLOSE
        )
        ahead_returns = roc.calculate(market_data)

        # Shifting ROC by time_ahead
        ahead_returns = ahead_returns.shift(-self.time_ahead)

        ahead_returns = ahead_returns.dropna()


        # Internal function to calculate the target labels
        def _calculate_target(ahead_return: float):

            result = HOLD  # Default signal (do nothing)

            if ahead_return - self.impact > self.buy_threshold:

                result = BUY  # Buy signal

            elif ahead_return + self.impact < self.sell_threshold:

                result = SELL  # Sell signal

            return result

        
        # Creating the targets dataframe        
        
        columns_tuples = [(symbol, 'TARGET') for symbol, _ in ahead_returns.columns]
        
        columns_multiindex = pd.MultiIndex.from_tuples(columns_tuples, names=['Symbols', 'Features'])

        targets = pd.DataFrame(0, index=ahead_returns.index, columns=columns_multiindex)        
        
        for symbol in ahead_returns.columns.levels[0]:

            targets.loc[:, (symbol, 'TARGET')] = ahead_returns[symbol]['AHEAD_RETURN'].apply(_calculate_target)

        return targets