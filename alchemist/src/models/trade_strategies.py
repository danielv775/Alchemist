
from abc import ABC, abstractmethod
import pandas as pd
from pandas.core.frame import DataFrame


from alchemist.src.consts import *

class TradeStrategy(ABC):

    # TODO replace start_value by portfolio at first date of trade_signals 
    # (CASH column will work as start_value and Quantity already held of the stock may be used for selling)
    @abstractmethod
    def calculate_trades(self, trades_signals: pd.DataFrame, market_data: pd.DataFrame, current_portfolio: DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def evaluate_strategy(market_data: DataFrame, trades: DataFrame):
        pass

class EvaluateClassifierStrategyMixin:

    def evaluate_strategy(market_data: DataFrame, trades: DataFrame):
        # TODO implement this method
        pass


class ClassifierStrategy(EvaluateClassifierStrategyMixin, TradeStrategy):

    def calculate_trades(self, trades_signals: DataFrame, market_data: DataFrame, current_portfolio: DataFrame) -> DataFrame:
        
        # Creating trades dataframe

        new_columns = [DELTA_HOLDING, DELTA_CASH]

        # Only needed for multi symbol TradeStrategies
        # symbols = trades_signals.columns.unique(level=SYMBOLS)
        # columns_tuples = [(symbol, new_column) for symbol in symbols for new_column in new_columns]
        symbol = trades_signals.columns.unique(level=SYMBOLS)[0] # Get the only symbol that should have in trades_signal
        columns_tuples = [(symbol, new_column) for new_column in new_columns]
        
        columns_multiindex = pd.MultiIndex.from_tuples(columns_tuples, names=[SYMBOLS, FEATURES])

        # Populating trades dataframe
       
        # initial_signals_date = trades_signals.index[0]  # First date of trades signal

        # # Get the Cash and Holding of the day closest to the first trade signal
        # current_cash = current_portfolio.loc[current_portfolio.index <= initial_signals_date, CASH].values[-1]
        # current_holding = current_portfolio.loc[current_portfolio.index <= initial_signals_date, symbol].values[-1]

        current_cash = current_portfolio[CASH].tail(1).values[-1]
        current_holding = current_portfolio[symbol].tail(1).values[-1]

        # Crop trade_signals to start at the last day of current portfolio dataframe
        trades_signals = trades_signals.loc[current_portfolio.index[-1]:, :]

        trades = pd.DataFrame(0.0, index=trades_signals.index, columns=columns_multiindex)

        for date, _ in trades_signals.iterrows():

            current_stock_price = market_data.loc[date, (symbol, ADJUSTED_CLOSE)]            

            # Evaluating signal to action (trades)
            if trades_signals.loc[date, (symbol, SIGNAL)] == BUY and current_cash > current_stock_price:

                # Go long  (buy most that we can)              
                
                stock_qtty = current_cash // current_stock_price
                stock_cost = -(current_stock_price * stock_qtty)  # TODO subtract fees                

                trades.loc[date, (symbol, DELTA_HOLDING)] = stock_qtty
                trades.loc[date, (symbol, DELTA_CASH)] = stock_cost
                

            elif trades_signals.loc[date, (symbol, SIGNAL)] == SELL and current_holding > 0:

                # Go short (sell all)               
                
                stock_cost = current_stock_price * current_holding  # TODO subtract fees                

                trades.loc[date, (symbol, DELTA_HOLDING)] = -current_holding
                trades.loc[date, (symbol, DELTA_CASH)] = stock_cost

            current_holding = trades.loc[date, (symbol, DELTA_HOLDING)] + current_holding
            current_cash = trades.loc[date, (symbol, DELTA_CASH)] + current_cash

            print(f'Current Holding: {current_holding}   Current cash: {current_cash}')

        return trades