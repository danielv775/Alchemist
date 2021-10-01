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


class ModelLayer(ABC):

    def __init__(self, model) -> None:
        self.model = model

    @abstractmethod
    def train(self, X: DataFrame, y: DataFrame):
        pass
    
    @abstractmethod
    def predict(self, X: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def evaluate_model(self, X: DataFrame, y: DataFrame, logger) -> DataFrame:
        pass


class EvaluateSkClassifierMixin:
    
    def evaluate_model(self, X: DataFrame, y: DataFrame, logger) -> DataFrame:
        
        from sklearn.metrics import confusion_matrix, plot_confusion_matrix

        y_hat = self.model.predict(X)        

        cf_matrix = confusion_matrix(y, y_hat, labels=[BUY, HOLD, SELL])
        logger.info('Confusion Matrix')
        logger.info(cf_matrix)
        logger.info('')

        plt.ioff()
        plot_confusion_matrix(self.model, X, y, labels=[BUY, HOLD, SELL])

        path = os.path.join(logger.name, 'confusion_matrix.png')
        plt.savefig(path)

        print('Ahoi')





class SkClassifierModelLayer(EvaluateSkClassifierMixin, ModelLayer):

    def train(self, X: DataFrame, y: DataFrame):
        
        self.model.fit(X, y)        


    def predict(self, X: DataFrame) -> DataFrame:
        
        y_hat = self.model.predict(X)

        return y_hat





        





        




