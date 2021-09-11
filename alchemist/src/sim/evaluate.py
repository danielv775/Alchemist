
from alchemist.src.strategies.basic import BasicTrader, HODLer
from alchemist.src.strategies.model import SKTrader
from alchemist.src.sim.marketsim import MarketSim
import numpy as np
import os

from alchemist.src.consts import *

from alchemist.src.etl.data_loader import load_market_data

import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from datetime import datetime

def evaluate_portfolio(portfolio: DataFrame) -> (float, float, float, float):

    """ Evaluate a portfolio with key metrics (cumulative return, mean daily return, std daily return, sharpe ratio)

    Args:
        portfolio (dataframe): dataframe with portfolio values

    Returns:
        cumulative_return (float): 
        mean_daily_return (float): 
        std_daily_return (float): 
        sharpe_ratio (float): 

    """

    cumulative_return = np.round((portfolio.values[-1]/portfolio.values[0]), 2)
    daily_returns = (portfolio.values[1:]/portfolio.values[:-1]) - 1

    mean_daily_return = np.round(daily_returns.mean(), 5)
    std_daily_return = np.round(daily_returns.std(), 5)
    sharpe_ratio = np.round((len(portfolio)**0.5) * (mean_daily_return/std_daily_return), 2)

    return cumulative_return, mean_daily_return, std_daily_return, sharpe_ratio

def print_portfolio_metrics(portfolio: DataFrame):

    """ Print portfolio metrics
    """
    
    cumulative_return, mean_daily_return, std_daily_return, sharpe_ratio = evaluate_portfolio(portfolio)

    print(f'Cumulative Return: {cumulative_return}')
    print(f'Mean Daily Return: {mean_daily_return}')
    print(f'Std Daily Return: {std_daily_return}')
    print(f'Sharpe Ratio (Annualized): {sharpe_ratio}')
    
    print(f'Start: {portfolio.values[0]}')
    print(f'End: {portfolio.values[-1]}')
    print(f'$Profit: {portfolio.values[-1] - portfolio.values[0]}')

def plot_trades(portfolio: DataFrame, trades: DataFrame, title: str, fp: str=f'{os.environ["PYTHONPATH"]}/alchemist/src/sim/graphs'):

    """ Plot trades as vertical lines on price graph. Green=Buy, Red=Sell

    Args:
        portfolio (dataframe):
        trades (dataframe):
        title (str):
        fp (str):
    """

    sd = trades.index[0]
    ed = trades.index[-1]

    symbol = trades.columns[0]

    data = load_market_data(
    [symbol], start_date=sd, end_date=ed, return_dict=False, invalidate_cache=False)
    
    price = data[(symbol, ADJUSTED_CLOSE)]

    actions = trades[trades[symbol] != 0.0]
    for a in actions.iterrows():
        
        timestamp = a[0]
        asset_change = a[1][symbol]

        if asset_change > 0:
            plt.axvline(timestamp, c='green')
        else:
            plt.axvline(timestamp, c='red')
    
    plt.plot(price.index, price.values, label='price')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=30)
    plt.savefig(f'{fp}/{title}.png')
    plt.close()


def plot_portfolios(portfolio: DataFrame, title: str, fp: str=f'{os.environ["PYTHONPATH"]}/alchemist/src/sim/graphs'):

    """ Plot multiple portfolio against one another
    """

    for strategy, portfolio in portfolios.items():
        plt.plot(portfolio.index, portfolio.values, label=strategy)
    
    plt.title('Portfolio Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend(loc='best')
    plt.xticks(rotation=30)
    plt.savefig(f'{fp}/{title}.png')
    plt.close()
        
if __name__ == '__main__':

    # Trading Decisions 
    trader = BasicTrader('Basic')
    trades = trader.trade(symbol='SQ')

    model = DecisionTreeRegressor(max_depth=5)
    dt_trader = SKTrader(model, 'DT', 'SQ')
    dt_trader.train(start_date=datetime(2020, 3, 1), end_date=datetime(2020, 5, 31))
    dt_trades = dt_trader.trade(start_date=datetime(2020, 6, 1), end_date=datetime(2020, 12, 31))

    hodler = HODLer('HODL')
    hodler_trades = hodler.trade(symbol='SQ')

    # Impact on Portfolio over time
    sim = MarketSim()
    portfolio_trader = sim.calculate_portfolio(trades)
    portfolio_hodler = sim.calculate_portfolio(hodler_trades)
    portfolio_dt = sim.calculate_portfolio(dt_trades)

    # Evaluate
    print('Basic Trader')
    print_portfolio_metrics(portfolio_trader['USD_Total'])

    print('..............................................')

    print('HODLer')
    print_portfolio_metrics(portfolio_hodler['USD_Total'])

    print('..............................................')

    print('DT')
    print_portfolio_metrics(portfolio_dt['USD_Total'])

    print('..............................................')

    portfolios = {
        'Trader': portfolio_trader['USD_Total'],
        'HODLer': portfolio_hodler['USD_Total'],
        'DT': portfolio_dt['USD_Total']
    }

    plot_portfolios(portfolios, title='compare')

    plot_trades(portfolio_trader, trades, title='Basic Trades')
    plot_trades(portfolio_dt, dt_trades, title='DT Trades')



