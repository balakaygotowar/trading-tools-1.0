"""
Backtesting and Analysis Module for MFTR Trading System
Handles probability analysis, parameter optimization, and performance evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import BACKTEST_CONFIG, ANALYSIS_CONFIG

class Backtester:
    """Class for backtesting trading strategies and analyzing performance"""
    
    def __init__(self, config=None):
        """Initialize with backtesting configuration"""
        self.config = config or BACKTEST_CONFIG
        self.analysis_config = ANALYSIS_CONFIG
    
    def analyze_event_probability(self, df, min_adx=None, min_angle_event=None, look_forward_bars=None):
        """
        Analyze probability of profitable trades based on signal criteria
        
        Args:
            df (pd.DataFrame): DataFrame with MFTR data and signals
            min_adx (float): Minimum ADX threshold
            min_angle_event (float): Minimum angle threshold
            look_forward_bars (int): Number of bars to look ahead for analysis
            
        Returns:
            dict: Analysis results including win rate and average returns
        """
        # Use config defaults if parameters not provided
        min_adx = min_adx or self.analysis_config['probability_analysis']['min_adx']
        min_angle_event = min_angle_event or self.analysis_config['probability_analysis']['min_angle_event']
        look_forward_bars = look_forward_bars or self.analysis_config['probability_analysis']['look_forward_bars']
        
        print(f"\n--- Running Probability Analysis ---")
        print(f"Event: Crossover with ADX > {min_adx} and Angle > {min_angle_event}")
        print(f"Analyzing price change over the next {look_forward_bars} bars.")
        
        # Find buy signals
        buy_signals = df[(df['mftrLine'] > df['mftrSignal']) & 
                        (df['mftrLine'].shift(1) < df['mftrSignal'].shift(1))]
        valid_buy_signals = buy_signals[(buy_signals['adxValue'] > min_adx) & 
                                       (buy_signals['angle'] > min_angle_event)]
        
        if valid_buy_signals.empty:
            print("No valid buy signals found for the specified criteria.")
            return {
                'signals_found': 0,
                'avg_change': 0,
                'win_rate': 0,
                'price_changes': []
            }
        
        price_changes = []
        valid_signals = []
        
        for index, row in valid_buy_signals.iterrows():
            start_index = df.index.get_loc(index)
            if start_index + look_forward_bars < len(df):
                entry_price = row['close']
                exit_price = df.iloc[start_index + look_forward_bars]['close']
                percent_change = ((exit_price - entry_price) / entry_price) * 100
                price_changes.append(percent_change)
                valid_signals.append({
                    'date': index,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': percent_change,
                    'adx': row['adxValue'],
                    'angle': row['angle']
                })
        
        if not price_changes:
            print("No valid signals with sufficient forward data.")
            return {
                'signals_found': 0,
                'avg_change': 0,
                'win_rate': 0,
                'price_changes': []
            }
        
        # Calculate statistics
        average_change = np.mean(price_changes)
        win_rate = (np.sum(np.array(price_changes) > 0) / len(price_changes)) * 100
        
        print(f"\nFound {len(valid_buy_signals)} valid buy signals.")
        print(f"Average price change after {look_forward_bars} bars: {average_change:.2f}%")
        print(f"Win Rate (price was higher after {look_forward_bars} bars): {win_rate:.2f}%")
        
        return {
            'signals_found': len(valid_buy_signals),
            'valid_for_analysis': len(price_changes),
            'avg_change': average_change,
            'win_rate': win_rate,
            'price_changes': price_changes,
            'signal_details': valid_signals,
            'statistics': {
                'max_return': max(price_changes),
                'min_return': min(price_changes),
                'std_return': np.std(price_changes),
                'median_return': np.median(price_changes)
            }
        }
    
    def optimize_angle_parameter(self, df_original, min_adx=None, stop_loss_pct=None, take_profit_pct=None, angle_range=None):
        """
        Optimize the minimum angle parameter using backtesting
        
        Args:
            df_original (pd.DataFrame): Original DataFrame with MFTR data
            min_adx (float): Minimum ADX threshold
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
            angle_range (range): Range of angles to test
            
        Returns:
            dict: Optimization results
        """
        # Use config defaults if parameters not provided
        min_adx = min_adx or self.analysis_config['parameter_optimization']['min_adx']
        stop_loss_pct = stop_loss_pct or self.analysis_config['parameter_optimization']['stop_loss_pct']
        take_profit_pct = take_profit_pct or self.analysis_config['parameter_optimization']['take_profit_pct']
        angle_range = angle_range or self.config['angle_range']
        
        print(f"\n--- Running Parameter Optimization for 'minAngle' ---")
        results = {}
        detailed_results = []
        
        for test_angle in angle_range:
            df = df_original.copy()
            wins = 0
            losses = 0
            trades = []
            
            # Find buy signals for this angle threshold
            buy_signals = df[(df['mftrLine'] > df['mftrSignal']) & 
                            (df['mftrLine'].shift(1) < df['mftrSignal'].shift(1))]
            valid_buy_signals = buy_signals[(buy_signals['adxValue'] > min_adx) & 
                                           (buy_signals['angle'] > test_angle)]
            
            for index, row in valid_buy_signals.iterrows():
                entry_price = row['close']
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                take_profit_price = entry_price * (1 + take_profit_pct / 100)
                start_iloc = df.index.get_loc(index)
                
                trade_result = None
                exit_price = None
                exit_reason = None
                
                # Check each subsequent bar for stop loss or take profit
                for i in range(start_iloc + 1, len(df)):
                    future_low = df.iloc[i]['low']
                    future_high = df.iloc[i]['high']
                    
                    if future_low <= stop_loss_price:
                        losses += 1
                        trade_result = 'loss'
                        exit_price = stop_loss_price
                        exit_reason = 'stop_loss'
                        break
                    if future_high >= take_profit_price:
                        wins += 1
                        trade_result = 'win'
                        exit_price = take_profit_price
                        exit_reason = 'take_profit'
                        break
                
                if trade_result:
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    trades.append({
                        'entry_date': index,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return_pct': return_pct,
                        'result': trade_result,
                        'exit_reason': exit_reason,
                        'adx': row['adxValue'],
                        'angle': row['angle']
                    })
            
            total_trades = wins + losses
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            results[test_angle] = win_rate
            
            detailed_results.append({
                'angle': test_angle,
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'trades': trades
            })
            
            print(f"Testing Angle: {test_angle}, Found {total_trades} trades, Win Rate: {win_rate:.2f}%")
        
        # Create visualization
        if results:
            self._plot_optimization_results(results)
        
        return {
            'results': results,
            'detailed_results': detailed_results,
            'best_angle': max(results.keys(), key=lambda k: results[k]) if results else None,
            'best_win_rate': max(results.values()) if results else 0
        }
    
    def _plot_optimization_results(self, results):
        """Create visualization for optimization results"""
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values(), width=2)
        plt.xlabel("Minimum Angle Setting")
        plt.ylabel("Win Rate (%)")
        plt.title("Optimization Results for Minimum Angle")
        plt.xticks(list(results.keys()))
        plt.grid(axis='y', linestyle='--')
        plt.show()
    
    def calculate_performance_metrics(self, trades):
        """
        Calculate comprehensive performance metrics from trade results
        
        Args:
            trades (list): List of trade dictionaries
            
        Returns:
            dict: Performance metrics
        """
        if not trades:
            return {'error': 'No trades provided'}
        
        returns = [trade['return_pct'] for trade in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades) * 100,
            'avg_return': np.mean(returns),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'best_trade': max(returns),
            'worst_trade': min(returns),
            'total_return': sum(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        }
        
        # Risk metrics
        if losses:
            metrics['max_drawdown'] = min(losses)
            metrics['profit_factor'] = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
        else:
            metrics['max_drawdown'] = 0
            metrics['profit_factor'] = float('inf')
        
        return metrics
    
    def run_full_analysis(self, df):
        """
        Run complete analysis including probability and optimization
        
        Args:
            df (pd.DataFrame): DataFrame with MFTR data
            
        Returns:
            dict: Complete analysis results
        """
        print("Running full MFTR analysis...")
        
        # Probability analysis
        prob_results = self.analyze_event_probability(df)
        
        # Parameter optimization
        opt_results = self.optimize_angle_parameter(df)
        
        return {
            'probability_analysis': prob_results,
            'parameter_optimization': opt_results,
            'data_summary': {
                'total_bars': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'avg_adx': df['adxValue'].mean(),
                'avg_angle': df['angle'].mean()
            }
        }

# Convenience functions for backward compatibility
def analyze_event_probability(df, min_adx=20, min_angle_event=40, look_forward_bars=20):
    """Convenience function for probability analysis"""
    backtester = Backtester()
    return backtester.analyze_event_probability(df, min_adx, min_angle_event, look_forward_bars)

def optimize_angle_parameter(df_original, min_adx=20, stop_loss_pct=2.0, take_profit_pct=4.0):
    """Convenience function for parameter optimization"""
    backtester = Backtester()
    return backtester.optimize_angle_parameter(df_original, min_adx, stop_loss_pct, take_profit_pct)