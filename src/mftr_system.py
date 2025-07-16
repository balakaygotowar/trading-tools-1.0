"""
Complete MFTR Trading System
Orchestrates all components for comprehensive trading analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from .advanced_data_fetcher import AdvancedDataFetcher
from .advanced_indicators import AdvancedIndicators, DEFAULT_MFTR_PARAMS
from .probability_analyzer import ProbabilityAnalyzer
from .parameter_optimizer import ParameterOptimizer


class MFTRSystem:
    """
    Complete MFTR Trading System
    Integrates data fetching, indicator calculation, analysis, and optimization
    """
    
    def __init__(self, exchange: str = 'coinbase'):
        """Initialize MFTR system with specified exchange"""
        self.data_fetcher = AdvancedDataFetcher(exchange)
        self.indicators = AdvancedIndicators()
        self.analyzer = ProbabilityAnalyzer()
        self.optimizer = ParameterOptimizer()
        self.df = None
        self.df_mftr = None
        
    def fetch_data(self, symbol: str = 'BTC/USDT', 
                   timeframe: str = '1h', 
                   total_bars: int = 5000) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for analysis
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            total_bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        self.df = self.data_fetcher.fetch_paginated_data(symbol, timeframe, total_bars)
        return self.df
    
    def calculate_indicators(self, params: Dict[str, Any] = None) -> Optional[pd.DataFrame]:
        """
        Calculate MFTR indicators on fetched data
        
        Args:
            params: MFTR parameters (uses defaults if None)
            
        Returns:
            DataFrame with MFTR indicators
        """
        if self.df is None:
            print("No data available. Please fetch data first.")
            return None
            
        if params is None:
            params = DEFAULT_MFTR_PARAMS.copy()
            
        self.df_mftr = self.indicators.calculate_mftr(self.df.copy(), **params)
        return self.df_mftr
    
    def run_probability_analysis(self, min_adx: float = 20, 
                               min_angle: float = 40, 
                               look_forward_bars: int = 24) -> Dict:
        """
        Run probability analysis on MFTR signals
        
        Args:
            min_adx: Minimum ADX for valid signals
            min_angle: Minimum angle for valid signals
            look_forward_bars: Bars to look forward
            
        Returns:
            Analysis results dictionary
        """
        if self.df_mftr is None:
            print("No MFTR data available. Please calculate indicators first.")
            return {}
            
        return self.analyzer.analyze_event_probability(
            self.df_mftr, min_adx, min_angle, look_forward_bars
        )
    
    def run_parameter_optimization(self, min_adx: float = 20,
                                 stop_loss_pct: float = 2.0,
                                 take_profit_pct: float = 4.0,
                                 angle_range: range = None) -> Dict[int, float]:
        """
        Optimize angle parameter for MFTR signals
        
        Args:
            min_adx: Minimum ADX for valid signals
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            angle_range: Range of angles to test
            
        Returns:
            Dictionary mapping angles to win rates
        """
        if self.df_mftr is None:
            print("No MFTR data available. Please calculate indicators first.")
            return {}
            
        results = self.optimizer.optimize_angle_parameter(
            self.df_mftr, min_adx, stop_loss_pct, take_profit_pct, angle_range
        )
        
        # Plot results
        if results:
            self.optimizer.plot_optimization_results(
                results, 
                "MFTR Angle Parameter Optimization",
                "Minimum Angle Setting",
                "Win Rate (%)"
            )
            
        return results
    
    def plot_mftr_signals(self, start_date: str = None, end_date: str = None) -> None:
        """
        Plot MFTR indicators with signals
        
        Args:
            start_date: Start date for plotting (YYYY-MM-DD)
            end_date: End date for plotting (YYYY-MM-DD)
        """
        if self.df_mftr is None:
            print("No MFTR data available. Please calculate indicators first.")
            return
            
        df_plot = self.df_mftr.copy()
        
        if start_date:
            df_plot = df_plot[df_plot.index >= start_date]
        if end_date:
            df_plot = df_plot[df_plot.index <= end_date]
            
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Plot 1: Price and signals
        ax1.plot(df_plot.index, df_plot['close'], label='Close Price', linewidth=1)
        
        # Mark buy signals
        buy_signals = df_plot[
            (df_plot['mftrLine'] > df_plot['mftrSignal']) & 
            (df_plot['mftrLine'].shift(1) < df_plot['mftrSignal'].shift(1)) &
            (df_plot['adxValue'] > 20) & (df_plot['angle'] > 40)
        ]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], 
                   color='green', marker='^', s=100, label='Buy Signals', zorder=5)
        ax1.set_ylabel('Price')
        ax1.set_title('MFTR Trading System - Price and Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MFTR lines
        ax2.plot(df_plot.index, df_plot['mftrLine'], label='MFTR Line', linewidth=1)
        ax2.plot(df_plot.index, df_plot['mftrSignal'], label='MFTR Signal', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('MFTR Values')
        ax2.set_title('MFTR Indicator Lines')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: ADX and Angle
        ax3_twin = ax3.twinx()
        ax3.plot(df_plot.index, df_plot['adxValue'], label='ADX', color='purple', linewidth=1)
        ax3_twin.plot(df_plot.index, df_plot['angle'], label='Angle', color='orange', linewidth=1)
        
        ax3.axhline(y=20, color='purple', linestyle='--', alpha=0.5, label='ADX Threshold')
        ax3_twin.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Angle Threshold')
        
        ax3.set_ylabel('ADX', color='purple')
        ax3_twin.set_ylabel('Angle', color='orange')
        ax3.set_xlabel('Date')
        ax3.set_title('ADX and Angle Indicators')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, symbol: str = 'BTC/USDT',
                            timeframe: str = '1h',
                            total_bars: int = 5000,
                            params: Dict[str, Any] = None) -> Dict:
        """
        Run complete MFTR analysis pipeline
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            total_bars: Number of bars to fetch
            params: MFTR parameters
            
        Returns:
            Complete analysis results
        """
        print("=== Starting Complete MFTR Analysis ===\n")
        
        # Step 1: Fetch data
        print("Step 1: Fetching data...")
        if self.fetch_data(symbol, timeframe, total_bars) is None:
            return {"error": "Failed to fetch data"}
        
        # Step 2: Calculate indicators
        print("\nStep 2: Calculating MFTR indicators...")
        if self.calculate_indicators(params) is None:
            return {"error": "Failed to calculate indicators"}
        
        # Step 3: Probability analysis
        print("\nStep 3: Running probability analysis...")
        prob_results = self.run_probability_analysis()
        
        # Step 4: Parameter optimization
        print("\nStep 4: Running parameter optimization...")
        opt_results = self.run_parameter_optimization()
        
        # Step 5: Generate plots
        print("\nStep 5: Generating visualization...")
        self.plot_mftr_signals()
        
        results = {
            "data_points": len(self.df_mftr),
            "probability_analysis": prob_results,
            "optimization_results": opt_results,
            "summary": {
                "total_signals": prob_results.get('total_signals', 0),
                "win_rate": prob_results.get('win_rate', 0),
                "average_return": prob_results.get('average_return', 0),
                "best_angle": max(opt_results, key=opt_results.get) if opt_results else None,
                "best_angle_performance": max(opt_results.values()) if opt_results else None
            }
        }
        
        print("\n=== Analysis Complete ===")
        print(f"Data points analyzed: {results['data_points']}")
        print(f"Total signals found: {results['summary']['total_signals']}")
        print(f"Overall win rate: {results['summary']['win_rate']:.1f}%")
        print(f"Average return: {results['summary']['average_return']:.2f}%")
        
        if results['summary']['best_angle']:
            print(f"Best angle parameter: {results['summary']['best_angle']} "
                  f"({results['summary']['best_angle_performance']:.1f}% win rate)")
        
        return results


# Convenience function for quick analysis
def run_mftr_analysis(symbol: str = 'BTC/USDT', 
                     timeframe: str = '1h', 
                     total_bars: int = 5000) -> Dict:
    """
    Convenience function to run complete MFTR analysis
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        total_bars: Number of bars to fetch
        
    Returns:
        Complete analysis results
    """
    system = MFTRSystem()
    return system.run_complete_analysis(symbol, timeframe, total_bars)