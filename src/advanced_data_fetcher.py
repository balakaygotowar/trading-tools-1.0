"""
Advanced Data Fetcher with Pagination Support
Robust data fetching from cryptocurrency exchanges with rate limiting
"""

import pandas as pd
import ccxt
import time
from typing import Optional


class AdvancedDataFetcher:
    """Advanced data fetcher with pagination and error handling"""
    
    def __init__(self, exchange_name: str = 'coinbase'):
        """Initialize with specified exchange"""
        self.exchange = getattr(ccxt, exchange_name)()
        
    def fetch_paginated_data(self, symbol: str = 'BTC/USDT', 
                           timeframe: str = '1h', 
                           total_bars: int = 5000) -> Optional[pd.DataFrame]:
        """
        Fetch historical data with pagination support
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data (1h, 4h, 1d)
            total_bars: Total number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        print(f"Fetching {total_bars} bars of {symbol} data for the {timeframe} timeframe from {self.exchange.name}...")
        
        try:
            limit_per_request = 300
            all_ohlcv = []
            timeframe_duration_in_ms = self.exchange.parse_timeframe(timeframe) * 1000
            since = self.exchange.milliseconds() - total_bars * timeframe_duration_in_ms
            
            while len(all_ohlcv) < total_bars:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit_per_request
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + timeframe_duration_in_ms
                
                print(f"Fetched {len(all_ohlcv)} bars so far...")
                time.sleep(self.exchange.rateLimit / 1000)
                
            # Convert to DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.drop_duplicates(subset='time', inplace=True)
            df.set_index('time', inplace=True)
            
            print(f"\nData fetched successfully! Total bars collected: {len(df)}")
            return df.sort_index()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None