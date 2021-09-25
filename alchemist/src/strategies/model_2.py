import pandas as pd


from datetime import datetime
import datetime as dt

from alchemist.src.strategies.trader import Trader
from pandas.core.frame import DataFrame

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np

import matplotlib.pyplot as plt

from alchemist.src.etl.data_loader import load_market_data
from alchemist.src.etl.technical_indicators import PriceBySMA, BBP, ROC
from alchemist.src.consts import *
from alchemist.src.sim.marketsim import MarketSim
from alchemist.src.sim import evaluate

class RandomForestTrader():

    def __init__(self, model,
                 name: str,
                 symbol: str,
                 impact: float = 0.0,
                 train_start_date: datetime = datetime(2019, 1, 1),
                 train_end_date: datetime = datetime(2019, 12, 31),
                 test_start_date: datetime = datetime(2020, 1, 1),
                 test_end_date: datetime = datetime(2020, 12, 31),
                 val_start_date: datetime = datetime(2021, 1, 1),
                 val_end_date: datetime = datetime(2021, 9, 17),                 
                 ):

        self.name = name
        self.impact = impact
        
        self.model = model        
        self.symbol = symbol
        
        self.days_ahead = 1
        self.buy_threshold = 0.02
        self.sell_threshold = -0.02

        # Loading daily market data
        market_data = load_market_data(
            [symbol, 'SPY'],
            train_start_date,
            val_end_date,
            return_dict=False,
            invalidate_cache=True
        )

        # Fill missing data
        market_data.fillna(method="ffill", inplace=True)
        market_data.fillna(method="bfill", inplace=True)

        market_data = market_data.drop(columns='SPY')[symbol]

        self.market_train_data = market_data.loc[train_start_date:train_end_date]
        self.market_test_data = market_data.loc[test_start_date:test_end_date]
        self.market_val_data = market_data.loc[val_start_date:val_end_date]

        self.train_data = self.get_features_and_target(self.market_train_data)
        self.test_data = self.get_features_and_target(self.market_test_data)
        self.val_data = self.get_features_and_target(self.market_val_data)



    def get_features_and_target(self, market_data: DataFrame):

        features_df = self.calculate_features(market_data)

        target_df = self.calculate_target(market_data)

        result = pd.concat([features_df, target_df], axis=1, join='inner')

        return result
       

    def calculate_features(self, market_data: DataFrame) -> DataFrame:
        
        # Calculating Features
        # Create instance of Price/SMA calculator
        feature_1_name = 'Price_by_SMA_25d'
        price_by_sma = PriceBySMA(
            name=feature_1_name, 
            window_size=25, 
            price_column_name=ADJUSTED_CLOSE
            )
        market_data = price_by_sma.calculate(market_data, merge_with_market_data=True)

        # Create instance of BBP calculator
        feature_2_name = 'BBP_25d'
        bbp = BBP(
            name=feature_2_name,
            window_size=25,
            price_column_name=ADJUSTED_CLOSE
            )
        market_data = bbp.calculate(market_data, merge_with_market_data=True)

        market_data = market_data.dropna()

        return market_data[[feature_1_name, feature_2_name]]

    
    def calculate_target(self, market_data: DataFrame) -> DataFrame:

        # Calculating Target

        roc = ROC(
            name=f'ROC_{self.days_ahead}d',
            window_size=self.days_ahead,
            price_column_name=ADJUSTED_CLOSE
        )
        market_data = roc.calculate(market_data, merge_with_market_data=True)

        # Shifting ROC by days_ahead (calculating n_day_returns)
        market_data[roc.name] = market_data[roc.name].shift(-self.days_ahead)

        market_data = market_data.dropna()

        # Internal function to calculate classification target
        def _calculate_target(n_day_ret: float):

            result = 0  # Default signal (do nothing)

            if n_day_ret - self.impact > self.buy_threshold:

                result = +1  # Buy signal

            elif n_day_ret + self.impact < self.sell_threshold:

                result = -1  # Sell signal

            return result

        # Buy signals
        market_data['TARGET'] = market_data[roc.name].apply(_calculate_target)

        return market_data[['TARGET']]

    def train(self):

        X = self.train_data.drop(columns='TARGET')

        y = self.train_data['TARGET']

        self.model.fit(X, y)

        y_pred = self.model.predict(X)

        n_correct = sum(y_pred == y)

        accuracy = n_correct / len(y)

        return accuracy


    def test(self):

        X = self.test_data.drop(columns='TARGET')

        y = self.test_data['TARGET']        

        y_pred = self.model.predict(X)

        n_correct = sum(y_pred == y)

        accuracy = n_correct / len(y)

        return accuracy


    def trade(self,            
              start_value: float = 10_000,
              ) -> DataFrame:

        
        X = self.calculate_features(self.market_val_data)

        y_pred = self.model.predict(X)

        trades = pd.DataFrame(0.0, index=self.market_val_data.index, columns=[self.symbol, 'USD'])

        current_holding = 0.0
        current_cash = start_value

        for date, row in X.iterrows():

            current_loc = X.index.get_loc(date)

            # Evaluating signal to action (trades)
            if y_pred[current_loc] == 1 and current_cash > 0:

                # Go long  (buy most that we can)              
                price = self.market_val_data.loc[date, ADJUSTED_CLOSE]
                stock_qtty = current_cash // price
                stock_cost = -price * stock_qtty  # TODO subtract fees                

                trades.loc[date][self.symbol] = stock_qtty
                trades.loc[date]['USD']= stock_cost
                

            elif y_pred[current_loc] == -1 and current_holding > 0:

                # Go short (sell all)
                price = self.market_val_data.loc[date, ADJUSTED_CLOSE]
                
                stock_cost = price * current_holding  # TODO subtract fees                

                trades.loc[date][self.symbol] = -current_holding
                trades.loc[date]['USD']= stock_cost

            current_holding = trades.loc[date][self.symbol] + current_holding
            current_cash = trades.loc[date]['USD'] + current_cash
       
        return trades



def main():

    symbol = 'SQ'

    model = RandomForestClassifier(max_depth=10, n_estimators=100)
    
    rf_trader = RandomForestTrader(model, 'RandomForestClassifier', symbol)

    train_accuracy = rf_trader.train()

    test_accuracy = rf_trader.test()

    print(f'Train accuracy: {train_accuracy}  Test accuracy: {test_accuracy}')

    start_value = 10_000

    trades = rf_trader.trade(start_value=start_value)    

    # portfolio = compute_portvals(symbol, trades, start_val = start_value, commission=0, impact=0)
    portfolio = MarketSim().calculate_portfolio(trades, start_value)

    evaluate.print_portfolio_metrics(portfolio)



if __name__ == '__main__':
    
    main()


