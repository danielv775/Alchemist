
import pandas as pd
from pandas.core.frame import DataFrame

import numpy as np

from abc import ABC, abstractmethod

from typing import Tuple

from datetime import datetime
import datetime as dt

from enum import Enum

import matplotlib.pyplot as plt
from copy import deepcopy

from alchemist.src.etl.data_loader import load_market_data
from alchemist.src.etl.technical_indicators import PriceBySMA, BBP, ROC, TechnicalIndicator
from alchemist.src.consts import *
from alchemist.src.sim.marketsim import MarketSim
from alchemist.src.sim import evaluate
from alchemist.src.models.feature_processors import FeaturesProcessor
from alchemist.src.models.target_processors import TargetProcessor
from alchemist.src.models.model_layers import ModelLayer
from alchemist.src.models.trade_strategies import TradeStrategy
from alchemist.src.helpers.config_mgmt_utils import ResultsLogger

class Learner(ABC):

    def __init__(
        self, features_processor: FeaturesProcessor, 
        target_processor: TargetProcessor, 
        model_layer: ModelLayer, 
        trade_strategy: TradeStrategy, 
        results_logger:ResultsLogger
        ):

        self.features_processor = features_processor

        self.target_processor = target_processor

        self.model_layer = model_layer

        self.trade_strategy = trade_strategy

        self.results_logger = results_logger

        self.trained = False


    def _get_training_data(self, market_data: DataFrame) -> Tuple[DataFrame, DataFrame]:

        X = self.features_processor.calculate_features(market_data)
        X.dropna(inplace=True)

        y = self.target_processor.calculate_targets(market_data)
        y.dropna(inplace=True)

        common_days = X.index.intersection(y.index)

        X = X.loc[common_days, :]
        y = y.loc[common_days, :]

        return X, y


    def train_model(self, market_data: DataFrame):

        X, y = self._get_training_data(market_data)

        self.model_layer.train(X, y)

        self.trained = True


    @abstractmethod
    def _calculate_trade_signals(self, market_data: DataFrame) -> DataFrame:
        pass


    def evaluate_learner(self, market_data: DataFrame, initial_cash: float, phase: Phase):
        
        X, y = self._get_training_data(market_data)

        self.model_layer.evaluate_model(X, y, self.results_logger, phase)

        # TODO call evaluate strategy method


    def trade_1(self, market_data: DataFrame, current_portfolio: DataFrame) -> tuple[DataFrame, DataFrame]:

        trade_signals = self._calculate_trade_signals(market_data)

        trades = self.trade_strategy.calculate_trades(trade_signals, market_data, current_portfolio)

        current_portfolio_columns = list(current_portfolio.columns)

        trades_columns = list(trades.columns.unique(level=SYMBOLS))

        # updated_portfolio_columns = {*current_portfolio, *trades_columns}  # This way doesn't keep the order
        updated_portfolio_columns = list(dict.fromkeys([*current_portfolio_columns, *trades_columns]))

        updated_portfolio = pd.DataFrame(0.0, index=trades.index, columns=updated_portfolio_columns)

        # Populating rows of updated_portfolio with rows from current_portfolio
        updated_portfolio.loc[current_portfolio.index, :] = current_portfolio.loc[current_portfolio.index, :]
        # for column in current_portfolio_columns:
            # updated_portfolio.loc[current_portfolio.index, column] = current_portfolio.loc[current_portfolio.index, column]

        # Populating columns CASH with available cash and holdings for each symbol
        for symbol in trades_columns:
            updated_portfolio.loc[:, CASH] = updated_portfolio.loc[:, CASH] + trades.loc[:, (symbol, DELTA_CASH)]
            updated_portfolio.loc[:, symbol] = updated_portfolio.loc[:, symbol] + trades.loc[:, (symbol, DELTA_HOLDING)]

        # Updating the Portfolio with Cash + value of stocks
        updated_portfolio[PORT_VALUE] = updated_portfolio[CASH]
        

        return trades, updated_portfolio


    def trade(self, market_data: DataFrame, current_portfolio: DataFrame) -> tuple[DataFrame, DataFrame]:

        trade_signals = self._calculate_trade_signals(market_data)

        trades = self.trade_strategy.calculate_trades(trade_signals, market_data, current_portfolio)

        non_symbol_cols = [CASH, PORT_VALUE]
        current_port_symbols = current_portfolio.columns[~current_portfolio.columns.isin(non_symbol_cols)]

        trades_symbols = list(trades.columns.unique(level=SYMBOLS))
        

        updated_portfolio_symbols = {*current_port_symbols, *trades_symbols}
        # updated_portfolio_columns = list(dict.fromkeys([*current_port_symbols, *trades_symbols, *non_symbol_cols]))

        first_trade_date = trades.index[0]
        first_market_date = market_data.index[0]
        last_portfolio_date = current_portfolio.index[-1]
        start_date = max(min(first_trade_date, last_portfolio_date), first_market_date)
        dates_index = market_data.loc[market_data.index >= start_date].index
        updated_portfolio = pd.DataFrame(0.0, index=dates_index, columns=[*updated_portfolio_symbols, *non_symbol_cols])
        
        updated_portfolio.loc[current_portfolio.index, :] = current_portfolio.loc[current_portfolio.index, :]       

        # Populating columns CASH with available cash and holdings for each symbol
        for symbol in trades_symbols:
            updated_portfolio.loc[trades.index, CASH] += trades.loc[trades.index, (symbol, DELTA_CASH)]
            updated_portfolio.loc[trades.index, symbol] += trades.loc[trades.index, (symbol, DELTA_HOLDING)]


        updated_portfolio = updated_portfolio.cumsum()
        
        # Updating the Portfolio with Cash + value of stocks
        updated_portfolio[PORT_VALUE] = updated_portfolio[CASH]
        
        for symbol in updated_portfolio_symbols:
            updated_portfolio[PORT_VALUE] += updated_portfolio[symbol] * market_data.loc[updated_portfolio[symbol].index, (symbol, CLOSE)]

        return trades, updated_portfolio

class ClassifierLearner(Learner):

    def __init__(
        self, 
        symbol: str, 
        features_processor: FeaturesProcessor, 
        target_processor: TargetProcessor, 
        model_layer: ModelLayer, 
        trade_strategy: TradeStrategy, 
        results_logger: ResultsLogger,
        ):

        self.symbol = symbol

        super().__init__(features_processor, target_processor, model_layer, trade_strategy, results_logger)


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
        # columns_tuples = [(symbol, SIGNAL) for symbol in symbols]
        columns_tuples = [(self.symbol, SIGNAL)]
        
        columns_multiindex = pd.MultiIndex.from_tuples(columns_tuples, names=[SYMBOLS, FEATURES])

        multiindex_signals = pd.DataFrame(signals, index=X.index, columns=columns_multiindex)

        return multiindex_signals
