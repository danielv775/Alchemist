

import pandas as pd

from alchemist.src.strategies.basic import BasicTrader
from pandas.core.frame import DataFrame

from alchemist.src.etl.data_loader import load_market_data
from alchemist.src.consts import *

class MarketSim():

    def __init__(self):
        pass

    def calculate_portfolio(self, trades: DataFrame, start_value:float=10000.0) -> DataFrame:
        """ Simulate how trades affect the positions in the portfolio and their value over time

        Args:
            trades (DataFrame): trades made on each day, defined by change in any asset +/-
            start_value (float, optional): initial value in USD. Defaults to 10000.0.

        Returns:
            DataFrame: DataFrame containing position sizes and their value over time
        """

        symbols = [ s for s in trades.columns if s != 'USD' ]
        
        start_date = trades.index[0]
        end_date = trades.index[-1]

        # Can infer what interval data to get based on trades datetime index
        data = load_market_data(symbols, start_date, end_date, return_dict=False, invalidate_cache=False)
        prices = data.loc[:, (symbols, ADJUSTED_CLOSE)] 

        portfolio = trades.copy()
        portfolio.loc[start_date, 'USD'] = start_value + trades.loc[start_date, 'USD'] 

        portfolio = portfolio.cumsum()  
        
        prices[('USD', ADJUSTED_CLOSE)] = 1.0

        symbols_USD = [f'{s}_USD' for s in symbols if s != 'USD']
        symbols_USD.append('USD')
        asset_value = pd.DataFrame(prices[portfolio.columns].values * portfolio.values, index=portfolio.index, columns=symbols_USD)
        asset_value['USD_Total'] = asset_value.sum(axis=1) 

        cols = asset_value.columns.difference(portfolio.columns)
        portfolio = portfolio.merge(asset_value[cols], left_index=True, right_index=True)
              
        return portfolio


if __name__ == '__main__':
    
    # Trading Decisions 
    trader = BasicTrader('Basic')
    trades = trader.trade(symbol='SQ')

    # Impact on Portfolio over time
    sim = MarketSim()
    portfolio = sim.calculate_portfolio(trades)

