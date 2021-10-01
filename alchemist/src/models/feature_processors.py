from abc import ABC, abstractmethod


from datetime import datetime
import datetime as dt
import pandas as pd
from pandas.core.frame import DataFrame

import matplotlib.pyplot as plt
from copy import deepcopy

from alchemist.src.etl.data_loader import load_market_data
from alchemist.src.consts import *
from alchemist.src.etl.technical_indicators import TechnicalIndicator


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