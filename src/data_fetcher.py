"""
Market Data Collection Module for MFTR Trading System
Handles fetching historical data from cryptocurrency exchanges
"""

import pandas as pd
import numpy as np
import ccxt
import time
import os
from config import DATA_CONFIG, get_data_directory

class DataFetcher:
    """Class for fetching market data from exchanges"""
    
    def __init__(self, config=None):
        """Initialize DataFetcher with configuration"""
        self.config = config or DATA_CONFIG
        self.exchange = None
        self._setup_exchange()
    
    def _setup_exchange(self):
        """Setup exchange connection"""
        exchange_name = self.config.get('exchange', 'coinbase')
        if exchange_name == 'coinbase':
            self.exchange = ccxt.coinbase()
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
    
    def fetch_paginated_data(self, symbol=None, timeframe=None, total_bars=None):
        """
        Fetch historical OHLCV data from exchange using pagination
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe for data (e.g., '1h', '1d')
            total_bars (int): Number of bars to fetch
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
        """
        symbol = symbol or self.config['symbol']
        timeframe = timeframe or self.config['timeframe']
        total_bars = total_bars or self.config['total_bars']
        limit_per_request = self.config['limit_per_request']
        
        print(f"Fetching {total_bars} bars of {symbol} data for the {timeframe} timeframe from {self.config['exchange']}...")
        
        try:
            all_ohlcv = []
            timeframe_duration_in_ms = self.exchange.parse_timeframe(timeframe) * 1000
            since = self.exchange.milliseconds() - total_bars * timeframe_duration_in_ms
            
            while len(all_ohlcv) < total_bars:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_request)
                if not ohlcv: 
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + timeframe_duration_in_ms
                print(f"Fetched {len(all_ohlcv)} bars so far...")
                time.sleep(self.exchange.rateLimit / 1000)
            
            df = self._process_ohlcv_data(all_ohlcv)
            
            # Add symbol column to track asset ticker
            df['symbol'] = symbol
            
            # Save DataFrame to CSV file
            csv_filename = self.config.get('csv_filename', 'market_data.csv')
            data_dir = get_data_directory()
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, csv_filename)
            df.to_csv(csv_path)
            print(f"Data saved to CSV: {csv_path}")
            
            print(f"\nData fetched successfully! Total bars collected: {len(df)}")
            return df
            
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            return None
    
    def _process_ohlcv_data(self, ohlcv_data):
        """
        Process raw OHLCV data into pandas DataFrame
        
        Args:
            ohlcv_data (list): Raw OHLCV data from exchange
            
        Returns:
            pd.DataFrame: Processed data with datetime index
        """
        df = pd.DataFrame(ohlcv_data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.drop_duplicates(subset='time', inplace=True)
        df.set_index('time', inplace=True)
        return df.sort_index()
    
    def validate_data(self, df):
        """
        Validate fetched data for completeness and quality
        
        Args:
            df (pd.DataFrame): OHLCV data to validate
            
        Returns:
            dict: Validation results
        """
        if df is None or df.empty:
            return {'valid': False, 'message': 'Data is empty or None'}
        
        validation_results = {
            'valid': True,
            'total_bars': len(df),
            'missing_values': df.isnull().sum().sum(),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'duplicates': df.index.duplicated().sum()
        }
        
        # Check for missing values
        if validation_results['missing_values'] > 0:
            validation_results['valid'] = False
            validation_results['message'] = f"Found {validation_results['missing_values']} missing values"
        
        # Check for duplicates
        if validation_results['duplicates'] > 0:
            validation_results['valid'] = False
            validation_results['message'] = f"Found {validation_results['duplicates']} duplicate timestamps"
        
        return validation_results
    
    def get_data_info(self, df):
        """
        Get information about the dataset
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            dict: Dataset information
        """
        if df is None or df.empty:
            return {'error': 'Data is empty or None'}
        
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'frequency': pd.infer_freq(df.index),
            'price_range': {
                'high': df['high'].max(),
                'low': df['low'].min(),
                'latest_close': df['close'].iloc[-1]
            },
            'volume_stats': {
                'mean': df['volume'].mean(),
                'max': df['volume'].max(),
                'min': df['volume'].min()
            }
        }

# Convenience function for backward compatibility
def fetch_paginated_data(symbol='BTC/USDT', timeframe='1h', total_bars=5000):
    """
    Convenience function for fetching data with default configuration
    Also saves data to CSV file automatically
    """
    fetcher = DataFetcher()
    return fetcher.fetch_paginated_data(symbol, timeframe, total_bars)

if __name__ == "__main__":
    """Execute data fetching when run directly"""
    print("=== MFTR Data Fetcher ===")
    df = fetch_paginated_data()
    if df is not None:
        print(f"✅ Successfully fetched and saved {len(df)} bars of data")
    else:
        print("❌ Data fetching failed")