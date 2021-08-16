from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
import numpy as np
import datetime as dt

from pandas.core.frame import DataFrame

from alchemist.src.consts import *

class TechnicalIndicator(ABC):

    def __init__(self, name:str):
        """Init for Abstract Technical Indicator 

        Args:
            name (str): Indicator name (name that will appear in the result DataFrame column)
        """
        
        self.name = name
        
    
    def calculate(self, market_data:DataFrame, merge_with_market_data:bool=False) -> DataFrame:
        """Calculate technical indicator values

        Args:
            market_data (DataFrame): DataFrame with Market Data (either single index or multiindex for more than one stock)
            merge_with_market_data (bool, optional): True to return the calculated data with the input market_data. Defaults to False.

        Raises:
            ValueError: [description]
            ValueError: [description]

        Returns:
            DataFrame: Either DataFrame with indicator values or DataFrame with market_data + indicator_values (merge_with_market_data=True)
        """

        # market_data.axes[0] is a Datetime Index (column Date)
        # market_data.axes[1].value is either
        #   Index(['High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose'], dtype='object', name='Features')
        # or
        #   MultiIndex([('TSLA',     'High'),
        #     ('TSLA',      'Low'),
        #     ('TSLA',     'Open'),
        #     ('TSLA',    'Close'),
        #     ('TSLA',   'Volume'),
        #     ('TSLA', 'AdjClose'),
        #     ('AAPL',     'High'),
        #     ('AAPL',      'Low'),
        #     ('AAPL',     'Open'),
        #     ('AAPL',    'Close'),
        #     ('AAPL',   'Volume'),
        #     ('AAPL', 'AdjClose')],
        #    names=['Symbols', 'Features'])        

        result_df = None

        if type(market_data.axes[1]) == pd.MultiIndex:

            result_dict = {}

            symbols = {symbol for symbol, _ in market_data.axes[1].values}            

            # Validate if the indicator was not already calculated
            if self.name in {feature for _, feature in market_data.axes[1].values}:
                raise ValueError(f'{self.name} is already in market data. Check if you are calculating the indicator twice!')
            
            # Iterate the axes[1] to calculate the indicator for each stock symbol 
            for symbol in symbols:

                df = self._calculate_and_rename_column(market_data[symbol])               

                result_dict[symbol]= df


            result_df = pd.concat(result_dict, axis=1)

            result_df = result_df.rename_axis(DATE).rename_axis([SYMBOLS, FEATURES], axis='columns')            

        else:

            # Validate if the indicator was not already calculated
            if self.name in market_data.columns.values:
                raise ValueError(f'{self.name} is already in market data. Check if you are calculating the indicator twice')                        

            result_df = self._calculate_and_rename_column(market_data)  


        if merge_with_market_data:
            result_df = market_data.merge(result_df, left_index=True, right_index=True)

        return result_df
    

    def _calculate_and_rename_column(self, market_data:DataFrame):
        
        df = self._calculate(market_data)

        # Renaming the first column (the only one) to the indicator name
        df.rename(columns = {list(df)[0]: self.name}, inplace=True)

        return df      


    @abstractmethod
    def _calculate(self, single_symbol_data:DataFrame):
        """Implements the specified technical indicator calculation.
        This function must be implemented in the children classes.

        Args:
            single_symbol_data (DataFrame):  DataFrame with Market Data of a single stock symbol. For example:

            Features          High         Low        Open       Close       Volume    AdjClose
            Date                                                                               
            2019-01-02  251.210007  245.949997  245.979996  250.179993  126925200.0  239.459229
            2019-01-03  248.570007  243.669998  248.229996  244.210007  144140700.0  233.745056
            2019-01-04  253.110001  247.169998  247.589996  252.389999  142628800.0  241.574493            
            (...)

        """

        # Implement this method in the child class
        pass


class SMA(TechnicalIndicator):

    def __init__(self, name:str, window_size:int, price_column_name:str=ADJUSTED_CLOSE):

        self.price_column_name = price_column_name

        self.window_size = window_size
        
        super().__init__(name)

    
    def _calculate(self, single_symbol_data:DataFrame):        

        # Slice the dataframe with a list market_data[[self.price_column_name]] to return a DataFrame
        # instead of string like market_data[self.price_column_name] which returns a Series        
        prices = single_symbol_data[[self.price_column_name]]

        # Check if there is any missing values
        if prices.isnull().values.any():
            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')

        result = prices.rolling(self.window_size).mean()

        return result


class PriceBySMA(TechnicalIndicator):

    def __init__(self, name:str, window_size:int, price_column_name:str=ADJUSTED_CLOSE):

        self.price_column_name = price_column_name

        self.window_size = window_size
        
        super().__init__(name)

    
    def _calculate(self, single_symbol_data:DataFrame):        

        # Slice the dataframe with a list market_data[[self.price_column_name]] to return a DataFrame
        # instead of string like market_data[self.price_column_name] which returns a Series        
        prices = single_symbol_data[[self.price_column_name]]

        # Check if there is any missing values
        if prices.isnull().values.any():
            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')

        sma = prices.rolling(self.window_size).mean()

        result = prices / sma

        return result


class BBP(TechnicalIndicator):

    def __init__(self, name:str, window_size:int, price_column_name:str=ADJUSTED_CLOSE):

        self.price_column_name = price_column_name

        self.window_size = window_size
        
        super().__init__(name)

    
    def _calculate(self, single_symbol_data:DataFrame):  

        prices = single_symbol_data[[self.price_column_name]]

        sma = prices.rolling(self.window_size).mean()

        stdev = result = prices.rolling(self.window_size).std()

        lower_band = sma - 2 * stdev

        upper_band = sma + 2 * stdev

        result = (prices - lower_band) / (upper_band - lower_band)

        return result        


# TODO Create more indicators like SMA and PriceBySMA (just create a class with __init__ and _caluclate functions)


# Example of Technical Indicators usage
if __name__ == '__main__':

    from datetime import datetime

    from alchemist.src.etl.data_loader import load_market_data
    

    start = datetime(2019, 1, 1)
    end = datetime(2020, 12, 31)

    symbols = ['TSLA', 'AAPL']

    # Loading daily market data as DataFrame (return_dict=False)
    daily_market_data = load_market_data(symbols, start, end, invalidate_cache=False, return_dict=False)
    print(daily_market_data)

    # Create instance of SMA calculator
    sma = SMA(name='20_days_SMA', window_size=20, price_column_name=ADJUSTED_CLOSE)

    # SMA calculator will calculate SMAs for TSLA and AAPL
    only_sma = sma.calculate(daily_market_data)
    print(only_sma)

    # SMA calculator will calculate SMAs for TSLA and AAPL and will merge with input market data (merge_with_market_data=True)
    sma_and_market_data = sma.calculate(daily_market_data, merge_with_market_data=True)
    print(sma_and_market_data)


    # Create instance of Price/SMA calculator
    price_by_sma = PriceBySMA(name='Price_by_SMA_25d', window_size=25, price_column_name=ADJUSTED_CLOSE)
    price_by_sma_values = price_by_sma.calculate(sma_and_market_data, merge_with_market_data=True)
    print(price_by_sma_values)

    # Create instance of BBP calculator
    bbp = BBP(name='bbp_25d', window_size=25, price_column_name=ADJUSTED_CLOSE)
    bbp_values = bbp.calculate(price_by_sma_values, merge_with_market_data=True)
    print(bbp_values)


    # Loading market data as dictionary (return_dict=True)
    symbols = ['SPY']
    spy_market_data_dict = load_market_data(symbols, start, end, invalidate_cache=False, return_dict=True)    
    
    # SMA calculator will calculate SMAs for SPY (spy_market_data_dict['SPY'] is a DataFrame with single index)
    spy_sma = sma.calculate(spy_market_data_dict['SPY'])
    print(spy_sma)

    # SMA calculator will calculate SMAs for SPY and will merge with input market data (merge_with_market_data=True)
    spy_sma_and_market_data = sma.calculate(spy_market_data_dict['SPY'], merge_with_market_data=True)
    print(spy_sma_and_market_data)

    # ----------------------------------------------

    # Ways to slice market data
    daily_market_data = load_market_data(['TSLA', 'AAPL'], start, end, invalidate_cache=False, return_dict=False)

    # Market data indexes are:
    print(daily_market_data.axes[0])
    print(daily_market_data.axes[1])
    
    # Getting High values between 2019-01-03 and 2019-01-07
    result = daily_market_data['TSLA'][HIGH].loc['2019-01-03':'2019-01-08']
    print(result)

    # Getting open values for TSLA and AAPL    
    result = daily_market_data.loc[:, (slice(None), OPEN)]
    print(result)


    # Getting open values for TSLA and AAPL between 2019-01-03 and 2019-01-07
    result = daily_market_data.loc['2019-01-03':'2019-01-08', (slice(None), OPEN)]
    print(result)
