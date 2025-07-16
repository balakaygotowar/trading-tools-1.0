"""
Probability Analysis for Trading Signals
Event-based backtesting and forward-looking probability analysis
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class ProbabilityAnalyzer:
    """Analyze probability and performance of trading signals"""
    
    @staticmethod
    def analyze_event_probability(df: pd.DataFrame, 
                                min_adx: float = 20, 
                                min_angle_event: float = 40, 
                                look_forward_bars: int = 20) -> dict:
        """
        Analyze probability of positive returns after signal events
        
        Args:
            df: DataFrame with MFTR indicators
            min_adx: Minimum ADX value for valid signals
            min_angle_event: Minimum angle for valid signals
            look_forward_bars: Number of bars to look forward
            
        Returns:
            Dictionary with analysis results
        """
        if df is None or df.empty:
            return {}
            
        print(f"\n--- Running Probability Analysis ---")
        print(f"Event: Crossover with ADX > {min_adx} and Angle > {min_angle_event}")
        print(f"Analyzing price change over the next {look_forward_bars} bars.")
        
        # Identify buy signals (crossover events)
        buy_signals = df[
            (df['mftrLine'] > df['mftrSignal']) & 
            (df['mftrLine'].shift(1) < df['mftrSignal'].shift(1))
        ]
        
        # Filter for valid signals based on criteria
        valid_buy_signals = buy_signals[
            (buy_signals['adxValue'] > min_adx) & 
            (buy_signals['angle'] > min_angle_event)
        ]
        
        price_changes = []
        valid_trades = []
        
        for index, row in valid_buy_signals.iterrows():
            start_index = df.index.get_loc(index)
            
            # Ensure we have enough future data
            if start_index + look_forward_bars < len(df):
                entry_price = row['close']
                exit_price = df.iloc[start_index + look_forward_bars]['close']
                percent_change = ((exit_price - entry_price) / entry_price) * 100
                
                price_changes.append(percent_change)
                valid_trades.append({
                    'entry_time': index,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'percent_change': percent_change,
                    'adx': row['adxValue'],
                    'angle': row['angle']
                })
        
        if not price_changes:
            print("No valid buy signals found for the specified criteria.")
            return {
                'total_signals': 0,
                'average_return': 0,
                'win_rate': 0,
                'trades': []
            }
        
        # Calculate statistics
        average_change = np.mean(price_changes)
        win_rate = (np.sum(np.array(price_changes) > 0) / len(price_changes)) * 100
        median_change = np.median(price_changes)
        std_change = np.std(price_changes)
        
        results = {
            'total_signals': len(valid_buy_signals),
            'valid_trades': len(valid_trades),
            'average_return': average_change,
            'median_return': median_change,
            'std_return': std_change,
            'win_rate': win_rate,
            'best_trade': max(price_changes) if price_changes else 0,
            'worst_trade': min(price_changes) if price_changes else 0,
            'trades': valid_trades
        }
        
        print(f"\nFound {len(valid_buy_signals)} valid buy signals.")
        print(f"Average price change after {look_forward_bars} bars: {average_change:.2f}%")
        print(f"Median price change: {median_change:.2f}%")
        print(f"Standard deviation: {std_change:.2f}%")
        print(f"Win Rate (price was higher after {look_forward_bars} bars): {win_rate:.2f}%")
        print(f"Best trade: {max(price_changes):.2f}%" if price_changes else "N/A")
        print(f"Worst trade: {min(price_changes):.2f}%" if price_changes else "N/A")
        
        return results
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio for a series of returns
        
        Args:
            returns: List of percentage returns
            risk_free_rate: Risk-free rate (default 0)
            
        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) == 0:
            return 0.0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown from a series of prices
        
        Args:
            prices: List of prices
            
        Returns:
            Tuple of (max_drawdown_pct, start_idx, end_idx)
        """
        if not prices or len(prices) < 2:
            return 0.0, 0, 0
            
        prices_array = np.array(prices)
        cumulative_max = np.maximum.accumulate(prices_array)
        drawdown = (prices_array - cumulative_max) / cumulative_max
        
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find the start of this drawdown
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                start_idx = i
                break
                
        return abs(max_dd) * 100, start_idx, max_dd_idx