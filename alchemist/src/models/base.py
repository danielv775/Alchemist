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


class FeaturesProcessor:
    
    def __init__(self, techinical_indicators: list[TechnicalIndicator]):
        
        self.techinical_indicators = techinical_indicators


    def _fill_missing_market_data(self, market_data: DataFrame) -> DataFrame:
        
        no_spy_data = 'SPY' not in market_data.columns.levels[0]
        
        if no_spy_data:
            # Loading daily market data
            spy_market_data = load_market_data(
                ['SPY'],
                market_data.index[0],
                market_data.index[-1],
                return_dict=False,
                invalidate_cache=True
            )
            
            # Left join with spy market data
            market_data = spy_market_data.join(market_data)
        
        # Fill missing data
        market_data = market_data.fillna(method="ffill")
        market_data = market_data.fillna(method="bfill")

        if no_spy_data:
            market_data = market_data.drop(columns='SPY') 

        return market_data       


    def calculate_features(self, market_data: DataFrame) -> DataFrame:

        market_data = deepcopy(market_data)

        market_data = self._fill_missing_market_data(market_data)

        features = None        

        for techinical_indicator in self.techinical_indicators:

            feature = techinical_indicator.calculate(market_data)

            if features is None:
                features = feature
            
            else:
                # Outer join between all features dataframe and the new feature
                features = pd.concat([features, feature], axis=1)

        return features


class TargetProcessor(ABC):

    @abstractmethod
    def calculate_targets(self, market_data: DataFrame) -> DataFrame:
        pass
        

class ModelLayer(ABC):

    def __init__(self, model) -> None:
        self.model = model

    @abstractmethod
    def train(self, X: DataFrame, y: DataFrame):
        pass
    
    @abstractmethod
    def predict(self, X: DataFrame) -> DataFrame:
        pass


class TradeStrategy(ABC):

    @abstractmethod
    def trade(self, trades_signals: DataFrame, market_data: DataFrame, start_value: float) -> DataFrame:
        pass


class Learner(ABC):

    def __init__(self, features_processor: FeaturesProcessor, target_processor: TargetProcessor, model_layer: ModelLayer, trade_strategy: TradeStrategy):

        self.features_processor = features_processor

        self.target_processor = target_processor

        self.model_layer = model_layer

        self.trade_strategy = trade_strategy

        self.trained = False


    def train_model(self, market_data: DataFrame):

        X = self.features_processor.calculate_features(market_data)
        X.dropna(inplace=True)

        y = self.target_processor.calculate_targets(market_data)
        y.dropna(inplace=True)

        common_days = X.index.intersection(y.index)

        X = X.loc[common_days, :]
        y = y.loc[common_days, :]

        self.model_layer.train(X, y)

        self.trained = True        

    @abstractmethod
    def _calculate_trade_signals(self, market_data: DataFrame) -> DataFrame:
        pass


    def evaluate_strategy(self, market_data: DataFrame):
        pass
        #self.model_layer.model


    def calculate_trades(self, market_data: DataFrame, start_value: float):

        trade_signals = self._calculate_trade_signals(market_data)

        trades = self.trade_strategy.trade(trade_signals, market_data, start_value)

        return trades









        





        




