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

class SKTrader(Trader):

    def __init__(self, model, name: str, impact: float=0.0):
        super().__init__(name, impact)
        self.model = model

    def train(self, symbol: str, start_date: datetime=datetime(2019, 1, 1), end_date: datetime=datetime(2019, 12, 31), start_value: float=10000):
        """Train Model

        Args:
            symbol (str): asset ticker (i.e. TSLA, SQ)
            start_date (datetime): start date
            end_date (datetime): end date
            start_value (float): initial USD
        """
        
        # Loading daily market data
        daily_market_data = load_market_data(
        [symbol], start_date, end_date, return_dict=False, invalidate_cache=False)

        # Create instance of Price/SMA calculator
        price_by_sma = PriceBySMA(name='Price_by_SMA_15d', window_size=15, price_column_name=ADJUSTED_CLOSE)
        daily_market_data = price_by_sma.calculate(daily_market_data, merge_with_market_data=True)

        # Create instance of BBP calculator
        bbp = BBP(name='bbp_15d', window_size=15, price_column_name=ADJUSTED_CLOSE)
        data = bbp.calculate(daily_market_data, merge_with_market_data=True)[symbol]

        data['ror_7d'] = data[ADJUSTED_CLOSE].pct_change(periods=7)

        data['ror_7d_future_true'] = data['ror_7d'].shift(-7)

        data = data.dropna()

        self.model.fit(data[[ADJUSTED_CLOSE, 'Price_by_SMA_15d', 'bbp_15d']], data['ror_7d_future_true'])

        pred_returns_train = self.model.predict(data[[ADJUSTED_CLOSE, 'Price_by_SMA_15d', 'bbp_15d']])
        data['ror_7d_future_pred'] = pred_returns_train

    
    def trade(self, symbol: str, start_date: datetime=datetime(2020, 1, 1), end_date: datetime=datetime(2020, 12, 31), start_value: float=10000) -> DataFrame:
        
        # Loading daily market data
        daily_market_data = load_market_data(
        [symbol], start_date, end_date, return_dict=False, invalidate_cache=False)

        # Create instance of Price/SMA calculator
        price_by_sma = PriceBySMA(name='Price_by_SMA_15d', window_size=15, price_column_name=ADJUSTED_CLOSE)
        daily_market_data = price_by_sma.calculate(daily_market_data, merge_with_market_data=True)

        # Create instance of BBP calculator
        bbp = BBP(name='bbp_15d', window_size=15, price_column_name=ADJUSTED_CLOSE)
        data = bbp.calculate(daily_market_data, merge_with_market_data=True)[symbol]

        data['ror_7d'] = data[ADJUSTED_CLOSE].pct_change(periods=7)
        data['ror_7d_future_true'] = data['ror_7d'].shift(-7)

        data = data.dropna()

        pred_returns = self.model.predict(data[[ADJUSTED_CLOSE, 'Price_by_SMA_15d', 'bbp_15d']])

        data['ror_7d_future_pred'] = pred_returns

        trades = pd.DataFrame(0, index=data.index, columns=[symbol, 'USD'])

        trading_days = trades.index
        position_status = 'close'

        current_usd = start_value 
        current_units = 0
        
        for td in trading_days:

            price = data.loc[td, ADJUSTED_CLOSE]

            if position_status == 'close':

                # BUY
                if (data.loc[td, 'Price_by_SMA_15d'] < 0.95) | (data.loc[td, 'bbp_15d'] < 0.2):
                    units = current_usd/price
                    order_value = price * units

                    trades.loc[td, symbol] = units
                    trades.loc[td, 'USD'] = -1*order_value

                    current_usd -= order_value

                    current_units += units

                    position_status = 'open'    

            elif position_status == 'open':
                
                # SELL
                if (data.loc[td, 'Price_by_SMA_15d'] > 1.1) | (data.loc[td, 'bbp_15d'] >= 1.1):
                    order_value = price * current_units

                    trades.loc[td, symbol] = -1*current_units
                    trades.loc[td, 'USD'] = order_value

                    current_usd += order_value

                    current_units = 0

                    position_status = 'close'
        
        return trades


if __name__ == '__main__':
    
    model = DecisionTreeRegressor(max_depth=15)
    dt_trader = SKTrader(model, 'DT', impact=0.0)
    dt_trader.train('SQ')
    dt_trader.trade('SQ')

    sdf