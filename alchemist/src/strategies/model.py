import pandas as pd

from alchemist.src.etl.data_loader import load_market_data
from alchemist.src.etl.technical_indicators import PriceBySMA, BBP
from alchemist.src.consts import *

from datetime import datetime
import datetime as dt 

from alchemist.src.strategies.trader import Trader
from pandas.core.frame import DataFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np

import matplotlib.pyplot as plt

class SKTrader(Trader):

    def __init__(self, model, 
                 name: str, 
                 symbol: str,
                 impact: float=0.0, 
                 window: int=7, 
                 start_date: datetime=datetime(2019, 1, 1), 
                 end_date: datetime=datetime(2020, 12, 31),
                 ):

        super().__init__(name, impact)
        self.model = model
        self.window = window
        self.symbol = symbol

        # Loading daily market data
        data = load_market_data(
        [symbol], start_date, end_date, return_dict=False, invalidate_cache=False)

        # Create instance of Price/SMA calculator
        price_by_sma = PriceBySMA(name=f'Price_by_SMA_{self.window}d', window_size=self.window, price_column_name=ADJUSTED_CLOSE)
        data = price_by_sma.calculate(data, merge_with_market_data=True)

        # Create instance of BBP calculator
        bbp = BBP(name=f'bbp_{self.window}d', window_size=self.window, price_column_name=ADJUSTED_CLOSE)
        data = bbp.calculate(data, merge_with_market_data=True)[symbol]

        data['daily_ror'] = data[ADJUSTED_CLOSE].pct_change(periods=1)

        data['daily_ror_future'] = data['daily_ror'].shift(-1)

        self.data = data
        self.features = [
            price_by_sma.name, bbp.name
        ]
        self.target = 'daily_ror_future'

        # Populate after training and trading period
        self.train_data = None
        self.test_data = None
    

    def train(self, 
              start_date: datetime=datetime(2019, 1, 1), 
              end_date: datetime=datetime(2019, 12, 31), 
              start_value: float=10000):
        """Train Model

        Args:
            start_date (datetime): start date
            end_date (datetime): end date
            start_value (float): initial USD
        """
        train_data = self.data.loc[start_date:end_date].dropna()

        self.model.fit(train_data[self.features], train_data[self.target])

        pred_returns = self.model.predict(train_data[self.features])
        train_data[f'{self.target}_pred'] = pred_returns
        self.train_data = train_data

    def trade(self, 
            start_date: datetime=datetime(2020, 1, 1), 
            end_date: datetime=datetime(2020, 12, 31), 
            start_value: float=10000) -> DataFrame:
        
        test_data = self.data.loc[start_date:end_date].dropna()

        pred_returns = self.model.predict(test_data[self.features])
        test_data[f'{self.target}_pred'] = pred_returns
        
        self.test_data = test_data

        trades = pd.DataFrame(0, index=test_data.index, columns=[self.symbol, 'USD'])

        trading_days = trades.index
        position_status = 'close'

        current_usd = start_value 
        current_units = 0
        
        for td in trading_days:
            
            pred_ror = test_data.loc[td, f'{self.target}_pred']
            price = test_data.loc[td, ADJUSTED_CLOSE]

            if position_status == 'close':

                # BUY
                if pred_ror > 0.0:
                    units = current_usd/price
                    order_value = price * units

                    trades.loc[td, self.symbol] = units
                    trades.loc[td, 'USD'] = -1*order_value

                    current_usd -= order_value

                    current_units += units

                    position_status = 'open'    

            elif position_status == 'open':
                
                # SELL
                if pred_ror < 0.0:
                    order_value = price * current_units

                    trades.loc[td, self.symbol] = -1*current_units
                    trades.loc[td, 'USD'] = order_value

                    current_usd += order_value

                    current_units = 0

                    position_status = 'close'
        
        return trades
    
    def evaluate_model(self, metrics=[r2_score, mean_absolute_error, mean_squared_error], plot=False):
        
        datasets = {
            'train': self.train_data,
            'test': self.test_data      
        }

        results = {}

        for split, data in datasets.items():
            
            performance = {}
            for m in metrics:
                
                y_true = data[self.target]
                y_pred = data[f'{self.target}_pred']
                
                performance[m.__name__] = m(y_true, y_pred)
            
            performance['corr'] = data[ [ *self.features, *[self.target] ] ].corr()

            results[split] = performance
        
        if plot:
            plt.scatter(y_true, y_pred, label='ypred')
            plt.plot(y_true, y_true, c='red', label='ytrue')
            title = self.target
            plt.title(title)
            plt.xlabel('ytrue')
            plt.ylabel('ypred')
            plt.show()
            plt.close()

        return results                

if __name__ == '__main__':
    
    # model = DecisionTreeRegressor(max_depth=5)
    model = RandomForestRegressor(max_depth=10, n_estimators=200)
    dt_trader = SKTrader(model, 'DT', 'SQ', window=14)
    dt_trader.train(start_date=datetime(2019, 4, 1), end_date=datetime(2019, 5, 30))
    trades = dt_trader.trade(start_date=datetime(2019, 6, 1), end_date=datetime(2019, 7, 31))

    results = dt_trader.evaluate_model(plot=True)

    print(results)
