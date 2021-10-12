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

from alchemist.src.helpers.config_mgmt_utils import ResultsLogger


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
    def evaluate_model(self, X: DataFrame, y: DataFrame, results_logger: ResultsLogger, phase: Phase) -> DataFrame:
        pass


class EvaluateClassifierMixin:
    
    def evaluate_model(self, X: DataFrame, y: DataFrame, results_logger: ResultsLogger, phase: Phase) -> DataFrame:
        
        from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay

        labels = [BUY, HOLD, SELL]

        y_hat = self.model.predict(X)        

        cf_matrix = confusion_matrix(y, y_hat, labels=labels)
        results_logger.log(f'Confusion Matrix: {phase.value}')
        results_logger.log(cf_matrix)
        results_logger.log('')

        disp = ConfusionMatrixDisplay(cf_matrix, display_labels=labels)

        disp.plot(
            include_values=True,
            # cmap='viridis', 
            cmap=plt.cm.Blues,
            xticks_rotation='horizontal'
        )

        path = os.path.join(results_logger.current_subfolder, f'confusion_matrix_{phase.name}.png')
        plt.title(f'Confusion Matrix: {phase.value}')
        plt.savefig(path)        

        cf_matrix_normed = confusion_matrix(y, y_hat, labels=labels, normalize='true')
        results_logger.log(f'Normalized Confusion Matrix: {phase.value}')
        results_logger.log(cf_matrix_normed)
        results_logger.log('')

        # plt.ioff()
        # plot_confusion_matrix(self.model, X, y, labels=[BUY, HOLD, SELL])

        disp = ConfusionMatrixDisplay(cf_matrix_normed, display_labels=labels)

        disp.plot(
            include_values=True,
            # cmap='viridis', 
            cmap=plt.cm.Blues,
            xticks_rotation='horizontal'
        )

        path = os.path.join(results_logger.current_subfolder, f'normed_confusion_matrix_{phase.name}.png')
        plt.title(f'Normalized Confusion Matrix: {phase.value}')
        plt.savefig(path)
        

class SkClassifierModelLayer(EvaluateClassifierMixin, ModelLayer):

    def train(self, X: DataFrame, y: DataFrame):
        
        self.model.fit(X, y)        


    def predict(self, X: DataFrame) -> DataFrame:
        
        y_hat = self.model.predict(X)

        return y_hat





        





        




