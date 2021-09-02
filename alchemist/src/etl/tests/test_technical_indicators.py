from abc import abstractmethod
import unittest
import os

import pandas as pd
from pandas import DataFrame
from pandas._testing import assert_series_equal

from alchemist.src.consts import *
from alchemist.src.etl.technical_indicators import TechnicalIndicator, SMA, PriceBySMA, BBP, ROC, MACD


def check_series_values_equal(left_series, right_series):

    try:
        assert_series_equal(left_series, right_series, check_exact=False, check_names=False)

        result = True

    except AssertionError:

        result = False

    return result


def load_test_data(excel_file_name):
    
    market_data = pd.read_excel(excel_file_name)

    market_data = market_data.set_index(DATE)

    return market_data


class TestTechnicalIndicator(unittest.TestCase):

    def setUp(self):        
        
        self.market_data = load_test_data(f"{os.environ['PYTHONPATH']}/alchemist/src/etl/tests/technical_indicator_test_data.xlsx")        

    
    def tearDown(self):
        pass    

    
    def test_calculate(self):

        # Mock implementation of the abstract Technical Indicator
        class MockIndicator(TechnicalIndicator): 

            def __init__(self, name:str, price_column_name:str):

                self.price_column_name = price_column_name                
                
                super().__init__(name)                       
            
            def _calculate(self, single_symbol_data:DataFrame):

                prices = single_symbol_data[[self.price_column_name]]

                # Only sum 1 to column AdjClose                
                # prices.loc[:, 'AdjClose'] += 1
                prices = prices + 1

                return prices


        mock_indicator = MockIndicator(name='mock_plus_1', price_column_name='AdjClose')

        calculated_mock = mock_indicator.calculate(self.market_data)

        self.assertIsInstance(calculated_mock, DataFrame)

        self.assertTrue(check_series_values_equal(calculated_mock['mock_plus_1'], self.market_data['True_Plus_1']))

        # Check if result is merged with initial dataframe
        calculated_mock_full_df = mock_indicator.calculate(self.market_data, merge_with_market_data=True)
        self.assertTrue(check_series_values_equal(calculated_mock['mock_plus_1'], calculated_mock_full_df['mock_plus_1']))
        self.assertTrue(check_series_values_equal(self.market_data['AdjClose'], calculated_mock_full_df['AdjClose']))

        

class TestSMA(unittest.TestCase):

    def setUp(self):        
        
        self.market_data = load_test_data(f"{os.environ['PYTHONPATH']}/alchemist/src/etl/tests/sma_test_data.xlsx")        

        # Creating a dataframe with null values
        self.market_data_with_nulls = self.market_data.copy()
        # self.market_data_with_nulls['AdjClose'].loc['2019-01-03'] = None
        #self.market_data_with_nulls.loc['2019-01-03']['AdjClose'] = None
        self.market_data_with_nulls.loc['2019-01-03', 'AdjClose'] = None

    
    def tearDown(self):
        pass    

    
    def test_calculate(self):                

        # Testing if values are calculated correctly        
        sma = SMA(name='10_days_SMA', window_size=10, price_column_name=ADJUSTED_CLOSE)
        calculated_10_day_sma = sma.calculate(self.market_data)
        self.assertTrue(check_series_values_equal(calculated_10_day_sma['10_days_SMA'], self.market_data['True_10_day_SMA']))

        sma = SMA(name='20_days_SMA', window_size=20, price_column_name=ADJUSTED_CLOSE)
        calculated_20_day_sma = sma.calculate(self.market_data)
        self.assertTrue(check_series_values_equal(calculated_20_day_sma['20_days_SMA'], self.market_data['True_20_day_SMA']))        

        with self.assertRaises(ValueError):
            sma.calculate(self.market_data_with_nulls)


class TestPriceBySMA(unittest.TestCase):

    def setUp(self):        
        
        self.market_data = load_test_data(f"{os.environ['PYTHONPATH']}/alchemist/src/etl/tests/price_by_sma_test_data.xlsx")        

        # Creating a dataframe with null values
        self.market_data_with_nulls = self.market_data.copy()
        # self.market_data_with_nulls['AdjClose'].loc['2019-01-03'] = None
        #self.market_data_with_nulls.loc['2019-01-03']['AdjClose'] = None
        self.market_data_with_nulls.loc['2019-01-03', 'AdjClose'] = None

    
    def tearDown(self):
        pass    

    
    def test_calculate(self):                

        # Testing if values are calculated correctly        
        price_by_sma = PriceBySMA(name='10_days_Price_by_SMA', window_size=10, price_column_name=ADJUSTED_CLOSE)
        calculated_10_day_price_by_sma = price_by_sma.calculate(self.market_data)
        self.assertTrue(check_series_values_equal(calculated_10_day_price_by_sma['10_days_Price_by_SMA'], self.market_data['True_10_day_Price_by_SMA']))

        price_by_sma = PriceBySMA(name='20_days_Price_by_SMA', window_size=20, price_column_name=ADJUSTED_CLOSE)
        calculated_20_day_price_by_sma = price_by_sma.calculate(self.market_data)
        self.assertTrue(check_series_values_equal(calculated_20_day_price_by_sma['20_days_Price_by_SMA'], self.market_data['True_20_day_Price_by_SMA']))        

        with self.assertRaises(ValueError):
            price_by_sma.calculate(self.market_data_with_nulls)


class TestBBP(unittest.TestCase):

    def setUp(self):        
        
        self.market_data = load_test_data(f"{os.environ['PYTHONPATH']}/alchemist/src/etl/tests/bbp_test_data.xlsx")        

        # Creating a dataframe with null values
        self.market_data_with_nulls = self.market_data.copy()
        # self.market_data_with_nulls['AdjClose'].loc['2019-01-03'] = None
        #self.market_data_with_nulls.loc['2019-01-03']['AdjClose'] = None
        self.market_data_with_nulls.loc['2019-01-03', 'AdjClose'] = None

    
    def tearDown(self):
        pass    

    
    def test_calculate(self):                

        # Testing if values are calculated correctly        
        bbp = BBP(name='10_days_BBP', window_size=10, price_column_name=ADJUSTED_CLOSE)
        calculated_10_day_bbp = bbp.calculate(self.market_data)
        self.assertTrue(check_series_values_equal(calculated_10_day_bbp['10_days_BBP'], self.market_data['True_10_day_BBP']))

        bbp = BBP(name='20_days_BBP', window_size=20, price_column_name=ADJUSTED_CLOSE)
        calculated_20_day_bbp = bbp.calculate(self.market_data)
        self.assertTrue(check_series_values_equal(calculated_20_day_bbp['20_days_BBP'], self.market_data['True_20_day_BBP']))

        with self.assertRaises(ValueError):
            bbp.calculate(self.market_data_with_nulls)


class TestROC(unittest.TestCase):

    def setUp(self):        
        
        self.market_data = load_test_data(f"{os.environ['PYTHONPATH']}/alchemist/src/etl/tests/roc_test_data.xlsx")        

        # Creating a dataframe with null values
        self.market_data_with_nulls = self.market_data.copy()
        # self.market_data_with_nulls['AdjClose'].loc['2019-01-03'] = None
        #self.market_data_with_nulls.loc['2019-01-03']['AdjClose'] = None
        self.market_data_with_nulls.loc['2019-01-03', 'AdjClose'] = None

    
    def tearDown(self):
        pass    

    
    def test_calculate(self):                

        # Testing if values are calculated correctly        
        roc = ROC(name='10_days_ROC', window_size=10, price_column_name=ADJUSTED_CLOSE)
        calculated_10_day_roc = roc.calculate(self.market_data)
        self.assertTrue(check_series_values_equal(calculated_10_day_roc['10_days_ROC'], self.market_data['True_10_day_ROC']))

        roc = ROC(name='20_days_ROC', window_size=20, price_column_name=ADJUSTED_CLOSE)
        calculated_20_day_roc = roc.calculate(self.market_data)
        self.assertTrue(check_series_values_equal(calculated_20_day_roc['20_days_ROC'], self.market_data['True_20_day_ROC']))

        with self.assertRaises(ValueError):
            roc.calculate(self.market_data_with_nulls)      

class TestMACD(unittest.TestCase):

    def setUp(self):        
        
        self.market_data = load_test_data(f"{os.environ['PYTHONPATH']}/alchemist/src/etl/tests/macd_test_data.xlsx")        

        # Creating a dataframe with null values
        self.market_data_with_nulls = self.market_data.copy()
        # self.market_data_with_nulls['AdjClose'].loc['2019-01-03'] = None
        #self.market_data_with_nulls.loc['2019-01-03']['AdjClose'] = None
        self.market_data_with_nulls.loc['2019-01-03', 'AdjClose'] = None

    
    def tearDown(self):
        pass    

    
    def test_calculate(self):                

        # Testing if values are calculated correctly        
        macd = MACD(
            name='MACD',
            signal_name='MACD_SIGNAL', 
            short_window=12, 
            long_window=26, 
            signal_window=9,
            price_column_name=ADJUSTED_CLOSE,
            )
        calculated_macd = macd.calculate(self.market_data)
        
        # Due to a calculation difference in Pandas and Excel, the first 200 lines doesn't match well. Checking the remaining lines only.
        self.assertTrue(check_series_values_equal(calculated_macd['MACD'].iloc[200:], self.market_data['True_MACD'].iloc[200:]))
        self.assertTrue(check_series_values_equal(calculated_macd['MACD_SIGNAL'].iloc[200:], self.market_data['True_9_day_EMA'].iloc[200:]))

      
        with self.assertRaises(ValueError):
            macd.calculate(self.market_data_with_nulls)          

if  __name__ == '__main__':
    
    unittest.main()
