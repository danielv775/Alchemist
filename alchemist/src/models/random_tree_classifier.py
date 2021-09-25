from dotenv import main
import pandas as pd

from abc import ABC, abstractmethod


from datetime import datetime
import datetime as dt

from alchemist.src.strategies.trader import Trader
from pandas.core.frame import DataFrame

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from alchemist.src.etl.data_loader import load_market_data
from alchemist.src.etl.technical_indicators import PriceBySMA, BBP, ROC, TechnicalIndicator
from alchemist.src.consts import *
from alchemist.src.sim.marketsim import MarketSim
from alchemist.src.sim import evaluate
from alchemist.src.models.base import TargetProcessor, FeaturesProcessor, ModelLayer, Learner, TradeStrategy



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


class SkModelLayer(ModelLayer):

    def train(self, X: DataFrame, y: DataFrame):
        
        self.model.fit(X, y)        


    def predict(self, X: DataFrame) -> DataFrame:
        
        y_hat = self.model.predict(X)

        return y_hat


class SkTradeStrategy(TradeStrategy):

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


class SkLearner(Learner):

    def __init__(self, symbol, features_processor: FeaturesProcessor, target_processor: TargetProcessor, model_layer: ModelLayer, trade_strategy: TradeStrategy):

        self.symbol = symbol

        super().__init__(features_processor, target_processor, model_layer, trade_strategy)


    def _calculate_trade_signals(self, market_data: DataFrame) -> DataFrame:

        assert self.trained, "Model required to be trained before calling trade method."

        symbol_market_data = market_data.loc[:, (self.symbol, slice(None))]

        # Calculate features
        X = self.features_processor.calculate_features(symbol_market_data)
        X.dropna(inplace=True)
        
        # Calculate signals
        signals = self.model_layer.predict(X)
        
        # Add signals into a DataFrame with dates
        
        # Only needed for multi symbol learners
        # symbols = X.columns.unique(level=SYMBOLS)
        # columns_tuples = [(symbol, 'SIGNAL') for symbol in symbols]
        columns_tuples = [(self.symbol, 'SIGNAL')]
        
        columns_multiindex = pd.MultiIndex.from_tuples(columns_tuples, names=['Symbols', 'Features'])

        trade_signals = pd.DataFrame(signals, index=X.index, columns=columns_multiindex)

        return trade_signals


def main():

    train_start_date = datetime(2019, 1, 1)
    train_end_date = datetime(2019, 12, 31)
    test_start_date = datetime(2020, 1, 1)
    test_end_date = datetime(2020, 12, 31)
    val_start_date = datetime(2021, 1, 1)
    val_end_date = datetime(2021, 9, 17)

    # Loading daily market data
    market_data = load_market_data(
                ['NOM', 'TSLA', 'SPY'],
                train_start_date,
                val_end_date,
                return_dict=False,
                invalidate_cache=False
            )

    # Creating Features Layer
    price_by_sma = PriceBySMA(
        name='price_by_sma', 
        window_size=25, 
        price_column_name=ADJUSTED_CLOSE
        )
        
    bbp = BBP(
        name='bbp',
        window_size=25,
        price_column_name=ADJUSTED_CLOSE
        )  

    roc = ROC(
            name='roc',
            window_size=1,
            price_column_name=ADJUSTED_CLOSE
        )  

    technical_indicators = [price_by_sma, bbp, roc]

    features_processor = FeaturesProcessor(technical_indicators)

    # Creating target layer
    target_processor = ClassifierTargetProcessor(time_ahead=5, buy_threshold=0.03, sell_threshold=-0.03)

    # Creating model layer
    sk_model = RandomForestClassifier(max_depth=10, n_estimators=100)
    model_layer = SkModelLayer(model=sk_model)

    # Creating trade strategy
    trade_strategy = SkTradeStrategy()

    learner = SkLearner('TSLA', features_processor, target_processor, model_layer, trade_strategy)

    train_market_data = market_data.loc[train_start_date:train_end_date, ('TSLA', slice(None))]

    learner.train_model(train_market_data)

    val_market_data = market_data.loc[test_start_date:test_end_date, ('TSLA', slice(None))]

    # Another way to slice it
    # idx = pd.IndexSlice
    # val_market_data = market_data.loc[val_start_date:val_end_date, idx['TSLA', :]]

    trades = learner.calculate_trades(val_market_data, start_value=10_000)    

    print(trades)

if __name__ == '__main__':
    
    main()





        





        




