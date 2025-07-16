"""
Parameter Optimization for Trading Strategies
Systematic testing and optimization of strategy parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class ParameterOptimizer:
    """Optimize trading strategy parameters through systematic testing"""
    
    @staticmethod
    def optimize_angle_parameter(df: pd.DataFrame, 
                               min_adx: float = 20, 
                               stop_loss_pct: float = 2.0, 
                               take_profit_pct: float = 4.0,
                               angle_range: range = None) -> Dict[int, float]:
        """
        Optimize the minimum angle parameter for MFTR signals
        
        Args:
            df: DataFrame with MFTR indicators
            min_adx: Minimum ADX value for valid signals
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            angle_range: Range of angles to test
            
        Returns:
            Dictionary mapping angle values to win rates
        """
        if df is None or df.empty:
            return {}
            
        if angle_range is None:
            angle_range = range(10, 51, 5)
            
        print(f"\n--- Running Parameter Optimization for 'minAngle' ---")
        print(f"Stop Loss: {stop_loss_pct}%, Take Profit: {take_profit_pct}%")
        print(f"Testing angles from {min(angle_range)} to {max(angle_range)}")
        
        results = {}
        detailed_results = {}
        
        for test_angle in angle_range:
            # Identify buy signals
            buy_signals = df[
                (df['mftrLine'] > df['mftrSignal']) & 
                (df['mftrLine'].shift(1) < df['mftrSignal'].shift(1))
            ]
            
            # Filter for valid signals
            valid_buy_signals = buy_signals[
                (buy_signals['adxValue'] > min_adx) & 
                (buy_signals['angle'] > test_angle)
            ]
            
            wins = 0
            losses = 0
            trades = []
            
            for index, row in valid_buy_signals.iterrows():
                entry_price = row['close']
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                take_profit_price = entry_price * (1 + take_profit_pct / 100)
                
                start_iloc = df.index.get_loc(index)
                trade_result = None
                exit_price = None
                exit_reason = None
                
                # Look forward for stop loss or take profit
                for i in range(start_iloc + 1, len(df)):
                    future_low = df.iloc[i]['low']
                    future_high = df.iloc[i]['high']
                    
                    # Check stop loss first (conservative approach)
                    if future_low <= stop_loss_price:
                        losses += 1
                        trade_result = 'loss'
                        exit_price = stop_loss_price
                        exit_reason = 'stop_loss'
                        break
                        
                    # Check take profit
                    if future_high >= take_profit_price:
                        wins += 1
                        trade_result = 'win'
                        exit_price = take_profit_price
                        exit_reason = 'take_profit'
                        break
                
                if trade_result:
                    trades.append({
                        'entry_time': index,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'result': trade_result,
                        'exit_reason': exit_reason,
                        'pnl_pct': ((exit_price - entry_price) / entry_price) * 100
                    })
            
            total_trades = wins + losses
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            
            results[test_angle] = win_rate
            detailed_results[test_angle] = {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'trades': trades
            }
            
            print(f"Angle: {test_angle:2d}, Trades: {total_trades:3d}, "
                  f"Wins: {wins:3d}, Losses: {losses:3d}, Win Rate: {win_rate:5.1f}%")
        
        # Store detailed results for further analysis
        ParameterOptimizer._detailed_results = detailed_results
        
        return results
    
    @staticmethod
    def plot_optimization_results(results: Dict[int, float], 
                                title: str = "Parameter Optimization Results",
                                xlabel: str = "Parameter Value",
                                ylabel: str = "Win Rate (%)") -> None:
        """
        Plot optimization results
        
        Args:
            results: Dictionary of parameter values to metrics
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if not results:
            print("No results to plot.")
            return
            
        plt.figure(figsize=(12, 8))
        plt.bar(results.keys(), results.values(), width=2, alpha=0.7, edgecolor='black')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(list(results.keys()))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for param, value in results.items():
            plt.text(param, value + 0.5, f'{value:.1f}%', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        best_param = max(results, key=results.get)
        best_performance = results[best_param]
        print(f"\nBest parameter: {best_param} with {best_performance:.1f}% win rate")
    
    @staticmethod
    def optimize_multiple_parameters(df: pd.DataFrame,
                                   param_ranges: Dict[str, range],
                                   base_params: Dict[str, float]) -> Dict[Tuple, float]:
        """
        Optimize multiple parameters simultaneously (grid search)
        
        Args:
            df: DataFrame with MFTR indicators
            param_ranges: Dictionary of parameter names to ranges
            base_params: Base parameters for optimization
            
        Returns:
            Dictionary mapping parameter combinations to performance metrics
        """
        if df is None or df.empty:
            return {}
            
        print(f"\n--- Running Multi-Parameter Optimization ---")
        print(f"Parameters to optimize: {list(param_ranges.keys())}")
        
        results = {}
        total_combinations = np.prod([len(r) for r in param_ranges.values()])
        current_combination = 0
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        def generate_combinations(ranges, current_combo=[]):
            if not ranges:
                yield tuple(current_combo)
            else:
                for value in ranges[0]:
                    yield from generate_combinations(ranges[1:], current_combo + [value])
        
        for combo in generate_combinations(param_values):
            current_combination += 1
            
            # Create parameter dictionary for this combination
            test_params = base_params.copy()
            for i, param_name in enumerate(param_names):
                test_params[param_name] = combo[i]
            
            # Test this parameter combination
            if 'min_angle' in test_params and 'min_adx' in test_params:
                angle_results = ParameterOptimizer.optimize_angle_parameter(
                    df, 
                    min_adx=test_params['min_adx'],
                    stop_loss_pct=test_params.get('stop_loss_pct', 2.0),
                    take_profit_pct=test_params.get('take_profit_pct', 4.0),
                    angle_range=range(int(test_params['min_angle']), 
                                    int(test_params['min_angle']) + 1)
                )
                
                if angle_results:
                    performance = list(angle_results.values())[0]
                    results[combo] = performance
            
            if current_combination % 10 == 0:
                print(f"Progress: {current_combination}/{total_combinations} combinations tested")
        
        return results
    
    @staticmethod
    def get_detailed_results() -> Dict:
        """Get detailed results from the last optimization run"""
        return getattr(ParameterOptimizer, '_detailed_results', {})