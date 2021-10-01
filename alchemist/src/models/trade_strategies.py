
from abc import ABC, abstractmethod
import pandas as pd
from pandas.core.frame import DataFrame

from alchemist.src.consts import *

class TradeStrategy(ABC):

    # TODO replace start_value by portfolio at first date of trade_signals 
    # (CASH column will work as start_value and Quantity already held of the stock may be used for selling)
    @abstractmethod
    def trade(self, trades_signals: pd.DataFrame, market_data: pd.DataFrame, start_value: float) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def evaluate_strategy(market_data: DataFrame, trades: DataFrame):
        pass

class EvaluateClassifierStrategyMixin:

    def evaluate_strategy(market_data: DataFrame, trades: DataFrame):
        # TODO implement this method
        pass


class ClassifierStrategy(EvaluateClassifierStrategyMixin, TradeStrategy):

    def trade(self, trades_signals: DataFrame, market_data: DataFrame, start_value: float) -> DataFrame:
        
        # Creating trades dataframe

        new_columns = ['STOCKS', 'USD']

        # Only needed for multi symbol TradeStrategies
        # symbols = trades_signals.columns.unique(level=SYMBOLS)
        # columns_tuples = [(symbol, new_column) for symbol in symbols for new_column in new_columns]
        symbol = trades_signals.columns.unique(level=SYMBOLS)[0] # Get the only symbol that should have in trades_signal
        columns_tuples = [(symbol, new_column) for new_column in new_columns]
        
        columns_multiindex = pd.MultiIndex.from_tuples(columns_tuples, names=['Symbols', 'Features'])

        trades = pd.DataFrame(0.0, index=market_data.index, columns=columns_multiindex)


        # Populating trades dataframe

        current_holding = 0.0
        current_cash = start_value        

        for date, _ in trades_signals.iterrows():

            current_stock_price = market_data.loc[date, (symbol, ADJUSTED_CLOSE)]            

            # Evaluating signal to action (trades)
            if trades_signals.loc[date, (symbol, 'SIGNAL')] == BUY and current_cash > current_stock_price:

                # Go long  (buy most that we can)              
                
                stock_qtty = current_cash // current_stock_price
                stock_cost = -(current_stock_price * stock_qtty)  # TODO subtract fees                

                trades.loc[date, (symbol, 'STOCKS')] = stock_qtty
                trades.loc[date, (symbol, 'USD')] = stock_cost
                

            elif trades_signals.loc[date, (symbol, 'SIGNAL')] == SELL and current_holding > 0:

                # Go short (sell all)               
                
                stock_cost = current_stock_price * current_holding  # TODO subtract fees                

                trades.loc[date, (symbol, 'STOCKS')] = -current_holding
                trades.loc[date, (symbol, 'USD')] = stock_cost

            current_holding = trades.loc[date, (symbol, 'STOCKS')] + current_holding
            current_cash = trades.loc[date, (symbol, 'USD')] + current_cash

            print(f'Current Holding: {current_holding}   Current cash: {current_cash}')

        return trades