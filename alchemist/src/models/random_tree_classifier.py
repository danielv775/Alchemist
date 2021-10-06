
import sys
import os

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

from alchemist.src.models.feature_processors import FeaturesProcessor
from alchemist.src.models.target_processors import TargetProcessor, ClassifierTargetProcessor
from alchemist.src.models.model_layers import ModelLayer, SkClassifierModelLayer
from alchemist.src.models.trade_strategies import TradeStrategy, ClassifierStrategy
from alchemist.src.models.learners import Learner, ClassifierLearner

from alchemist.src.strategies.basic import HODLer

from alchemist.src.helpers.config_mgmt_utils import ResultsLogger


def main():

    results_logger = ResultsLogger()

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
    sk_model = RandomForestClassifier(max_depth=5, n_estimators=100)
    model_layer = SkClassifierModelLayer(model=sk_model)

    # Creating trade strategy
    trade_strategy = ClassifierStrategy()

    learner = ClassifierLearner('TSLA', features_processor, target_processor, model_layer, trade_strategy, results_logger)

    train_market_data = market_data.loc[train_start_date:train_end_date, ('TSLA', slice(None))]

    learner.train_model(train_market_data)

    # TODO Test model
    learner.evaluate_learner(train_market_data)

    val_market_data = market_data.loc[val_start_date:val_end_date, ('TSLA', slice(None))]

    # Another way to slice it
    # idx = pd.IndexSlice
    # val_market_data = market_data.loc[val_start_date:val_end_date, idx['TSLA', :]]
    
    trades = learner.calculate_trades(val_market_data, start_value=10_000)    
    print(trades)

    final_stock_price = market_data.loc[test_end_date, ('TSLA', ADJUSTED_CLOSE)]
    hodler = HODLer('abc')
    result = hodler.trade('TSLA', test_start_date, test_end_date, start_value=10_000)
    print(result)

    
    print(result.loc[:, 'TSLA'][0] * final_stock_price)


if __name__ == '__main__':
    
    main()





        





        




