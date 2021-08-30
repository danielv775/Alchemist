
from alchemist.src.strategies.basic import BasicTrader, HODLer
from alchemist.src.sim.marketsim import MarketSim
import numpy as np
import os

import matplotlib.pyplot as plt

def evaluate_portfolio(portfolio):

	cumulative_return = np.round((portfolio.values[-1]/portfolio.values[0]), 2)
	daily_returns = (portfolio.values[1:]/portfolio.values[:-1]) - 1

	mean_daily_return = np.round(daily_returns.mean(), 5)
	std_daily_return = np.round(daily_returns.std(), 5)
	sharpe_ratio = np.round((len(portfolio)**0.5) * (mean_daily_return/std_daily_return), 2)

	return cumulative_return, mean_daily_return, std_daily_return, sharpe_ratio

def print_portfolio_metrics(portfolio):
    
    cumulative_return, mean_daily_return, std_daily_return, sharpe_ratio = evaluate_portfolio(portfolio)

    print(f'Cumulative Return: {cumulative_return}')
    print(f'Mean Daily Return: {mean_daily_return}')
    print(f'Std Daily Return: {std_daily_return}')
    print(f'Sharpe Ratio (Annualized): {sharpe_ratio}')
    
def plot_portfolios(portfolios, fp=f'{os.environ["PYTHONPATH"]}/alchemist/src/sim/graphs'):

    for strategy, portfolio in portfolios.items():
        plt.plot(portfolio.index, portfolio.values, label=strategy)
    
    plt.title('Portfolio Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend(loc='best')
    plt.xticks(rotation=30)
    plt.savefig(f'{fp}/compare.png')
        
if __name__ == '__main__':
    
    # Trading Decisions 
    trader = BasicTrader('Basic')
    trades = trader.trade(symbol='SQ')    

    hodler = HODLer('HODL')
    hodler_trades = hodler.trade(symbol='SQ')

    # Impact on Portfolio over time
    sim = MarketSim()
    portfolio_trader = sim.calculate_portfolio(trades)
    portfolio_trader.to_csv('t.csv')

    portfolio_hodler = sim.calculate_portfolio(hodler_trades)

    # Evaluate
    print('Basic Trader')
    print_portfolio_metrics(portfolio_trader['USD_Total'])

    print('..............................................')

    print('HODLer')
    print_portfolio_metrics(portfolio_hodler['USD_Total'])

    portfolios = {
        'Trader': portfolio_trader['USD_Total'],
        'HODLer': portfolio_hodler['USD_Total']
    }

    plot_portfolios(portfolios)


