from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
import numpy as np
import datetime as dt

from pandas.core.frame import DataFrame

from alchemist.src.consts import *

class TechnicalIndicator(ABC):

    def __init__(self):
        """Init for Abstract Technical Indicator 
        """
        
        # self.name = name
        pass
        
    
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
            
            # Iterate the axes[1] to calculate the indicator for each stock symbol 
            for symbol in symbols:

                # df = self._calculate_and_rename_column(market_data[symbol])
                df = self._calculate(market_data[symbol].copy())

                result_dict[symbol]= df


            result_df = pd.concat(result_dict, axis=1)

            result_df = result_df.rename_axis(DATE).rename_axis([SYMBOLS, FEATURES], axis='columns')            

        else:                      

            # result_df = self._calculate_and_rename_column(market_data)
            result_df = self._calculate(market_data.copy())


        # Validate if the result columns already exist in original market data
        if result_df is not None and any([col_name in list(market_data) for col_name in list(result_df)]):
            raise ValueError(f'{self.name} is already in market data. Check if you are calculating the indicator twice')           


        if merge_with_market_data:
            result_df = market_data.merge(result_df, left_index=True, right_index=True)

        return result_df   


    @abstractmethod
    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:
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


        Returns:
            DataFrame: A dataframe whose first column will be renamed automatically by self.name
                The implementation of most of indicators should return only one column, unless the indicator
                has auxiliary curves like MACD or Bollinger Bands (the auxiliary curves should be 2nd column and after).
        """

        # Implement this method in the child class        


class SMA(TechnicalIndicator):

    def __init__(self, name:str, window_size:int, price_column_name:str=ADJUSTED_CLOSE):

        self.name = name

        self.price_column_name = price_column_name

        self.window_size = window_size
        
        super().__init__()

    
    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:        

        # Slice the dataframe with a list single_symbol_data[[self.price_column_name]] to return a DataFrame
        # instead of string like single_symbol_data[self.price_column_name] which returns a Series        
        prices = single_symbol_data[[self.price_column_name]]

        # Check if there is any missing values
        if prices.isnull().values.any():
            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')

        result = prices.rolling(self.window_size).mean()

        result.rename(columns = {self.price_column_name: self.name}, inplace=True)

        return result


class PriceBySMA(TechnicalIndicator):

    def __init__(self, name:str, window_size:int, price_column_name:str=ADJUSTED_CLOSE):

        self.name = name

        self.price_column_name = price_column_name

        self.window_size = window_size
        
        super().__init__()

    
    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:        

        # Slice the dataframe with a list market_data[[self.price_column_name]] to return a DataFrame
        # instead of string like market_data[self.price_column_name] which returns a Series        
        prices = single_symbol_data[[self.price_column_name]]

        # Check if there is any missing values
        if prices.isnull().values.any():
            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')

        sma = prices.rolling(self.window_size).mean()

        result = prices / sma

        result.rename(columns = {self.price_column_name: self.name}, inplace=True)

        return result


class BBP(TechnicalIndicator):
    # https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce

    def __init__(self, name:str, window_size:int, price_column_name:str=ADJUSTED_CLOSE):

        self.name = name

        self.price_column_name = price_column_name

        self.window_size = window_size
        
        super().__init__()

    
    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:  

        prices = single_symbol_data[[self.price_column_name]]

        # Check if there is any missing values
        if prices.isnull().values.any():
            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')        

        sma = prices.rolling(self.window_size).mean()

        stdev = result = prices.rolling(self.window_size).std()

        lower_band = sma - 2 * stdev

        upper_band = sma + 2 * stdev

        result = (prices - lower_band) / (upper_band - lower_band)

        result.rename(columns = {self.price_column_name: self.name}, inplace=True)

        return result        


class ROC(TechnicalIndicator):
    # https://www.investopedia.com/terms/p/pricerateofchange.asp

    def __init__(self, name:str, window_size:int, price_column_name:str=ADJUSTED_CLOSE):

        self.name = name

        self.price_column_name = price_column_name

        self.window_size = window_size
        
        super().__init__()

    
    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:  

        prices = single_symbol_data[[self.price_column_name]]

        # Check if there is any missing values
        if prices.isnull().values.any():
            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')        
        
        result = prices.pct_change(self.window_size, fill_method='ffill')

        result.rename(columns = {self.price_column_name: self.name}, inplace=True)

        return result  


class MACD(TechnicalIndicator):
    # https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
    # https://investexcel.net/how-to-calculate-macd-in-excel/
    # https://www.learnpythonwithrune.org/calucate-macd-with-pandas-dataframes/

    def __init__(
        self, 
        name:str, 
        signal_name:str, 
        short_window:int=12, 
        long_window:int=26, 
        signal_window:int=9, 
        price_column_name:str=ADJUSTED_CLOSE
        ):
        self.name = name
        self.signal_name = signal_name
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.price_column_name = price_column_name
        
        super().__init__()

    
    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:  

        prices = single_symbol_data[[self.price_column_name]]

        # Check if there is any missing values
        if prices.isnull().values.any():
            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')
        
        
        short_ema = prices.ewm(span=self.short_window, adjust=False).mean()

        long_ema = prices.ewm(span=self.long_window, adjust=False).mean()        

        # MACD
        result = short_ema - long_ema
        result.rename(columns = {self.price_column_name: self.name}, inplace=True)

        # Adding the signal line to the dataframe
        signal_ema = result.ewm(span=self.signal_window, adjust=False).mean()
        result[self.signal_name] = signal_ema

        return result


class EMA(TechnicalIndicator):

    def __init__(self, name:str, window_size:int, price_column_name:str=ADJUSTED_CLOSE):

        self.name = name

        self.price_column_name = price_column_name

        self.window_size = window_size
        
        super().__init__()

    
    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:        

        # Slice the dataframe with a list single_symbol_data[[self.price_column_name]] to return a DataFrame
        # instead of string like single_symbol_data[self.price_column_name] which returns a Series        
        prices = single_symbol_data[[self.price_column_name]]

        # Check if there is any missing values
        if prices.isnull().values.any():
            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')

        result = prices.ewm(span=self.window_size, adjust=False).mean()

        result.rename(columns = {self.price_column_name: self.name}, inplace=True)

        return result
        

class FastStochasticOscillator(TechnicalIndicator):
    # https://medium.com/codex/algorithmic-trading-with-stochastic-oscillator-in-python-7e2bec49b60d

    def __init__(
        self,
        fast_k_name: str = 'fast_k',
        fast_d_name: str = 'fast_d',
        k_window: int = 14,
        d_window: int = 3,
        low_column: str = LOW,
        high_column: str = HIGH,
        close_column: str = CLOSE
    ):
        self.fast_k_name = fast_k_name
        self.fast_d_name = fast_d_name
        self.k_window = k_window
        self.d_window = d_window
        self.low_column = low_column
        self.high_column = high_column
        self.close_column = close_column

        super().__init__()


    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:

        # Check if there is any missing values
        if single_symbol_data[self.low_column].isnull().values.any() or \
                single_symbol_data[self.high_column].isnull().values.any() or \
                single_symbol_data[self.close_column].isnull().values.any():

            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')

        low_min  = single_symbol_data[self.low_column].rolling(window = self.k_window).min()
        
        high_max = single_symbol_data[self.high_column].rolling(window = self.k_window).max()


        # Fast Stochastic
        fast_k = 100 * (single_symbol_data[self.close_column] - low_min)/(high_max - low_min)
        fast_d = fast_k.rolling(window=self.d_window).mean()

        result = pd.concat([fast_k, fast_d], axis=1)
        result.rename(columns = {0: self.fast_k_name, 1: self.fast_d_name}, inplace=True)

        return result



class SlowStochasticOscillator(TechnicalIndicator):
    # https://stackoverflow.com/questions/30261541/slow-stochastic-implementation-in-python-pandas

    def __init__(
        self,
        slow_k_name: str = 'slow_k',
        slow_d_name: str = 'slow_d',
        k_window: int = 14,
        d_window: int = 3,
        low_column: str = LOW,
        high_column: str = HIGH,
        close_column: str = CLOSE
    ):

        self.slow_k_name = slow_k_name
        self.slow_d_name = slow_d_name
        self.k_window = k_window
        self.d_window = d_window
        self.low_column = low_column
        self.high_column = high_column
        self.close_column = close_column

        super().__init__()


    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:

        # Check if there is any missing values
        if single_symbol_data[self.low_column].isnull().values.any() or \
                single_symbol_data[self.high_column].isnull().values.any() or \
                single_symbol_data[self.close_column].isnull().values.any():

            raise ValueError('Market Data contains null values. Missing preprocessing interpolation.')        

        low_min  = single_symbol_data[self.low_column].rolling(window = self.k_window).min()
        
        high_max = single_symbol_data[self.high_column].rolling(window = self.k_window).max()


        # Fast Stochastic
        fast_k = 100 * (single_symbol_data[self.close_column] - low_min)/(high_max - low_min)
        fast_d = fast_k.rolling(window=self.d_window).mean()

        # Slow Stochastic
        slow_k = fast_d
        slow_d = slow_k.rolling(window=self.d_window).mean()        

        result = pd.concat([slow_k, slow_d], axis=1)
        result.rename(columns = {0: self.slow_k_name, 1: self.slow_d_name}, inplace=True)

        return result


class ChaikinMoneyFlow(TechnicalIndicator):
    # https://www.tradingview.com/support/solutions/43000501974-chaikin-money-flow-cmf/
    # https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmf

    def __init__(
        self,
        name: str,
        window: int = 21,
        volume_column: str = VOLUME,
        low_column: str = LOW,
        high_column: str = HIGH,
        close_column: str = CLOSE
    ):

        self.name = name
        self.window = window
        self.volume_column = volume_column
        self.low_column = low_column
        self.high_column = high_column
        self.close_column = close_column

        super().__init__()


    def _calculate(self, single_symbol_data:DataFrame) -> DataFrame:
        
        # Check if there is any missing values
        if single_symbol_data[self.volume_column].isnull().values.any() or \
            single_symbol_data[self.low_column].isnull().values.any() or \
                single_symbol_data[self.high_column].isnull().values.any() or \
                single_symbol_data[self.close_column].isnull().values.any():

            raise ValueError(
                'Market Data contains null values. Missing preprocessing interpolation.')

        close = single_symbol_data[self.close_column]
        low = single_symbol_data[self.low_column]
        high = single_symbol_data[self.high_column]
        volume = single_symbol_data[self.volume_column]

        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)

        money_flow = money_flow_multiplier * volume

        cmf = money_flow.rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()

        result = pd.DataFrame({self.name: cmf})

        return result


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

    fso = FastStochasticOscillator('fast_k', 'fast_d')
    fso_indicator = fso.calculate(daily_market_data, merge_with_market_data=True)

    sso = SlowStochasticOscillator()
    sso_indicator = sso.calculate(daily_market_data, merge_with_market_data=True)


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
