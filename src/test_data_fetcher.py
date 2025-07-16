"""
Unit Tests for Data Fetcher Module
Tests data collection, validation, and processing functionality
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import ccxt

import sys
import os
# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher, fetch_paginated_data
from config import DATA_CONFIG

class TestDataFetcher(unittest.TestCase):
    """Test cases for DataFetcher class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DATA_CONFIG.copy()
        self.fetcher = DataFetcher(self.config)
        
        # Create sample OHLCV data
        self.sample_ohlcv = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],  # 2022-01-01 00:00
            [1640998800000, 47200.0, 47800.0, 47100.0, 47600.0, 1100.0],  # 2022-01-01 01:00
            [1641002400000, 47600.0, 48000.0, 47400.0, 47900.0, 1200.0],  # 2022-01-01 02:00
            [1641006000000, 47900.0, 48200.0, 47700.0, 48000.0, 1300.0],  # 2022-01-01 03:00
            [1641009600000, 48000.0, 48300.0, 47800.0, 48100.0, 1400.0],  # 2022-01-01 04:00
        ]
        
        # Create expected DataFrame
        self.expected_df = pd.DataFrame(
            self.sample_ohlcv, 
            columns=['time', 'open', 'high', 'low', 'close', 'volume']
        )
        self.expected_df['time'] = pd.to_datetime(self.expected_df['time'], unit='ms')
        self.expected_df.set_index('time', inplace=True)
    
    def test_initialization(self):
        """Test DataFetcher initialization"""
        # Test with default config
        fetcher = DataFetcher()
        self.assertIsNotNone(fetcher.config)
        self.assertIsNotNone(fetcher.exchange)
        
        # Test with custom config
        custom_config = {'exchange': 'coinbase', 'symbol': 'ETH/USDT'}
        fetcher = DataFetcher(custom_config)
        self.assertEqual(fetcher.config['symbol'], 'ETH/USDT')
    
    def test_setup_exchange(self):
        """Test exchange setup"""
        # Test coinbase exchange
        fetcher = DataFetcher({'exchange': 'coinbase'})
        self.assertIsInstance(fetcher.exchange, ccxt.coinbase)
        
        # Test unsupported exchange
        with self.assertRaises(ValueError):
            DataFetcher({'exchange': 'unsupported_exchange'})
    
    @patch('ccxt.coinbase')
    def test_fetch_paginated_data_success(self, mock_exchange_class):
        """Test successful data fetching"""
        # Mock exchange instance
        mock_exchange = Mock()
        mock_exchange_class.return_value = mock_exchange
        
        # Configure mock exchange
        mock_exchange.parse_timeframe.return_value = 3600  # 1 hour
        mock_exchange.milliseconds.return_value = 1641013200000  # Fixed timestamp
        mock_exchange.rateLimit = 1000
        mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv
        
        fetcher = DataFetcher()
        result = fetcher.fetch_paginated_data('BTC/USDT', '1h', 5)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        self.assertIsInstance(result.index, pd.DatetimeIndex)
    
    @patch('ccxt.coinbase')
    def test_fetch_paginated_data_empty_response(self, mock_exchange_class):
        """Test handling of empty response"""
        mock_exchange = Mock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.return_value = []
        
        fetcher = DataFetcher()
        result = fetcher.fetch_paginated_data('BTC/USDT', '1h', 5)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)
    
    @patch('ccxt.coinbase')
    def test_fetch_paginated_data_exception(self, mock_exchange_class):
        """Test exception handling during data fetch"""
        mock_exchange = Mock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.side_effect = Exception("Network error")
        
        fetcher = DataFetcher()
        result = fetcher.fetch_paginated_data('BTC/USDT', '1h', 5)
        
        self.assertIsNone(result)
    
    def test_process_ohlcv_data(self):
        """Test OHLCV data processing"""
        fetcher = DataFetcher()
        result = fetcher._process_ohlcv_data(self.sample_ohlcv)
        
        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        
        # Check data integrity
        self.assertEqual(result.iloc[0]['close'], 47200.0)
        self.assertEqual(result.iloc[-1]['close'], 48100.0)
        
        # Check sorting
        self.assertTrue(result.index.is_monotonic_increasing)
    
    def test_process_ohlcv_data_with_duplicates(self):
        """Test processing data with duplicate timestamps"""
        # Add duplicate timestamp
        duplicate_data = self.sample_ohlcv + [self.sample_ohlcv[0]]
        
        fetcher = DataFetcher()
        result = fetcher._process_ohlcv_data(duplicate_data)
        
        # Should remove duplicates
        self.assertEqual(len(result), 5)  # Original 5, not 6
    
    def test_validate_data_valid(self):
        """Test data validation with valid data"""
        fetcher = DataFetcher()
        validation = fetcher.validate_data(self.expected_df)
        
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['total_bars'], 5)
        self.assertEqual(validation['missing_values'], 0)
        self.assertEqual(validation['duplicates'], 0)
    
    def test_validate_data_empty(self):
        """Test data validation with empty data"""
        fetcher = DataFetcher()
        
        # Test None
        validation = fetcher.validate_data(None)
        self.assertFalse(validation['valid'])
        self.assertIn('empty', validation['message'])
        
        # Test empty DataFrame
        validation = fetcher.validate_data(pd.DataFrame())
        self.assertFalse(validation['valid'])
        self.assertIn('empty', validation['message'])
    
    def test_validate_data_missing_values(self):
        """Test data validation with missing values"""
        df_with_nan = self.expected_df.copy()
        df_with_nan.iloc[0, 0] = np.nan  # Add NaN to first row, first column
        
        fetcher = DataFetcher()
        validation = fetcher.validate_data(df_with_nan)
        
        self.assertFalse(validation['valid'])
        self.assertEqual(validation['missing_values'], 1)
        self.assertIn('missing values', validation['message'])
    
    def test_validate_data_duplicates(self):
        """Test data validation with duplicate indices"""
        df_with_duplicates = pd.concat([self.expected_df, self.expected_df.iloc[:1]])
        
        fetcher = DataFetcher()
        validation = fetcher.validate_data(df_with_duplicates)
        
        self.assertFalse(validation['valid'])
        self.assertEqual(validation['duplicates'], 1)
        self.assertIn('duplicate', validation['message'])
    
    def test_get_data_info_valid(self):
        """Test getting data information from valid DataFrame"""
        fetcher = DataFetcher()
        info = fetcher.get_data_info(self.expected_df)
        
        self.assertEqual(info['shape'], (5, 5))
        self.assertEqual(len(info['columns']), 5)
        self.assertIn('price_range', info)
        self.assertIn('volume_stats', info)
        self.assertEqual(info['price_range']['high'], 48300.0)
        self.assertEqual(info['price_range']['low'], 46800.0)
    
    def test_get_data_info_empty(self):
        """Test getting data information from empty DataFrame"""
        fetcher = DataFetcher()
        info = fetcher.get_data_info(None)
        
        self.assertIn('error', info)
        
        info = fetcher.get_data_info(pd.DataFrame())
        self.assertIn('error', info)

class TestDataQuality(unittest.TestCase):
    """Test cases for data quality validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.valid_data = pd.DataFrame({
            'time': pd.date_range('2022-01-01', periods=5, freq='H'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0],
            'low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }).set_index('time')
    
    def test_price_relationships(self):
        """Test that high >= close >= low for all rows"""
        for idx, row in self.valid_data.iterrows():
            self.assertGreaterEqual(row['high'], row['close'], f"High < Close at {idx}")
            self.assertGreaterEqual(row['close'], row['low'], f"Close < Low at {idx}")
            self.assertGreaterEqual(row['high'], row['low'], f"High < Low at {idx}")
    
    def test_price_reasonableness(self):
        """Test that prices are within reasonable ranges"""
        for col in ['open', 'high', 'low', 'close']:
            self.assertTrue((self.valid_data[col] > 0).all(), f"Negative prices in {col}")
            self.assertTrue((self.valid_data[col] < 1000000).all(), f"Unreasonably high prices in {col}")
    
    def test_volume_reasonableness(self):
        """Test that volume values are reasonable"""
        self.assertTrue((self.valid_data['volume'] >= 0).all(), "Negative volume")
    
    def test_chronological_order(self):
        """Test that timestamps are in chronological order"""
        self.assertTrue(self.valid_data.index.is_monotonic_increasing, "Timestamps not chronological")

class TestConvenienceFunction(unittest.TestCase):
    """Test cases for convenience function"""
    
    @patch('data_fetcher.DataFetcher')
    def test_fetch_paginated_data_function(self, mock_fetcher_class):
        """Test the convenience function"""
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher
        mock_fetcher.fetch_paginated_data.return_value = pd.DataFrame()
        
        result = fetch_paginated_data('BTC/USDT', '1h', 100)
        
        mock_fetcher_class.assert_called_once()
        mock_fetcher.fetch_paginated_data.assert_called_once_with('BTC/USDT', '1h', 100)
        self.assertIsInstance(result, pd.DataFrame)

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)