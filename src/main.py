"""
Main execution script for MFTR Trading System
Orchestrates all modules and provides command-line interface
"""

import sys
import argparse
import pandas as pd
import os
from data_fetcher import DataFetcher
from indicators import TechnicalIndicators
from signal_generator import SignalGenerator
from backtester import Backtester
from config import Config, get_data_directory

class MFTRSystem:
    """Main class for MFTR Trading System"""
    
    def __init__(self, config=None):
        """Initialize MFTR system with configuration"""
        self.config = config or Config()
        self.data_fetcher = DataFetcher(self.config.data_config)
        self.indicators = TechnicalIndicators(self.config.mftr_params)
        self.signal_generator = SignalGenerator(self.config.signal_config)
        self.backtester = Backtester(self.config.backtest_config)
        self.df_raw = None
        self.df_mftr = None
        self.df_signals = None
    
    def load_data_from_csv(self, csv_filename=None):
        """
        Load market data from CSV file
        
        Args:
            csv_filename (str): Name of CSV file to load
            
        Returns:
            bool: Success status
        """
        print("=== Loading Market Data from CSV ===")
        
        try:
            csv_filename = csv_filename or self.config.data_config.get('csv_filename', 'market_data.csv')
            data_dir = get_data_directory()
            csv_path = os.path.join(data_dir, csv_filename)
            
            if not os.path.exists(csv_path):
                print(f"❌ CSV file not found: {csv_path}")
                return False
            
            # Load CSV data
            self.df_raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            if self.df_raw is not None and not self.df_raw.empty:
                # Validate data
                validation = self.data_fetcher.validate_data(self.df_raw)
                if validation['valid']:
                    print("✅ CSV data loaded and validated successfully")
                    # Print data info
                    info = self.data_fetcher.get_data_info(self.df_raw)
                    print(f"📊 Data shape: {info['shape']}")
                    print(f"📅 Date range: {info['date_range']}")
                    print(f"💰 Price range: ${info['price_range']['low']:,.2f} - ${info['price_range']['high']:,.2f}")
                    return True
                else:
                    print(f"❌ CSV data validation failed: {validation.get('message', 'Unknown error')}")
                    return False
            else:
                print("❌ CSV file is empty or could not be loaded")
                return False
                
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            return False
    
    def fetch_data(self, symbol=None, timeframe=None, total_bars=None, use_csv=True):
        """
        Fetch market data - loads from CSV by default, falls back to API
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            total_bars (int): Number of bars to fetch
            use_csv (bool): Whether to load from CSV first (default: True)
            
        Returns:
            bool: Success status
        """
        if use_csv:
            # Try loading from CSV first
            if self.load_data_from_csv():
                return True
            else:
                print("⚠️ CSV loading failed, falling back to API fetch...")
        
        print("=== Fetching Market Data from API ===")
        
        try:
            self.df_raw = self.data_fetcher.fetch_paginated_data(symbol, timeframe, total_bars)
            
            if self.df_raw is not None:
                # Validate data
                validation = self.data_fetcher.validate_data(self.df_raw)
                if validation['valid']:
                    print("✅ Data validation passed")
                    # Print data info
                    info = self.data_fetcher.get_data_info(self.df_raw)
                    print(f"📊 Data shape: {info['shape']}")
                    print(f"📅 Date range: {info['date_range']}")
                    print(f"💰 Price range: ${info['price_range']['low']:,.2f} - ${info['price_range']['high']:,.2f}")
                    return True
                else:
                    print(f"❌ Data validation failed: {validation.get('message', 'Unknown error')}")
                    return False
            else:
                print("❌ Failed to fetch data")
                return False
                
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
    
    def calculate_indicators(self):
        """
        Calculate MFTR indicators
        
        Returns:
            bool: Success status
        """
        print("\n=== Calculating MFTR Indicators ===")
        
        if self.df_raw is None:
            print("❌ No raw data available. Please fetch data first.")
            return False
        
        try:
            self.df_mftr = self.indicators.calculate_mftr(self.df_raw.copy())
            
            if self.df_mftr is not None:
                print(f"✅ MFTR calculation completed")
                print(f"📊 Final dataset: {len(self.df_mftr)} rows")
                
                # Show sample of key indicators
                key_cols = ['close', 'mftrLine', 'mftrSignal', 'adxValue', 'angle']
                print("\n📈 Latest indicator values:")
                print(self.df_mftr[key_cols].tail(3))
                return True
            else:
                print("❌ MFTR calculation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return False
    
    def generate_signals(self):
        """
        Generate trading signals
        
        Returns:
            bool: Success status
        """
        print("\n=== Generating Trading Signals ===")
        
        if self.df_mftr is None:
            print("❌ No MFTR data available. Please calculate indicators first.")
            return False
        
        try:
            self.df_signals = self.signal_generator.generate_buy_signals(self.df_mftr)
            
            if self.df_signals is not None:
                # Get signal details
                signal_details = self.signal_generator.get_signal_details(self.df_signals)
                
                print(f"✅ Signal generation completed")
                print(f"🎯 Total crossovers: {signal_details['total_crossovers']}")
                print(f"🎯 Final buy signals: {signal_details['final_buy_signals']}")
                print(f"🎯 Filtering efficiency: {signal_details['filtering_efficiency']}")
                print(f"🎯 Signal frequency: {signal_details['signal_frequency']}")
                
                # Validate signals
                validation = self.signal_generator.validate_signals(self.df_signals)
                if validation['warnings']:
                    print("⚠️ Warnings:")
                    for warning in validation['warnings']:
                        print(f"  • {warning}")
                
                return True
            else:
                print("❌ Signal generation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            return False
    
    def run_analysis(self):
        """
        Run backtesting analysis
        
        Returns:
            dict: Analysis results
        """
        print("\n=== Running Analysis ===")
        
        if self.df_mftr is None:
            print("❌ No MFTR data available. Please calculate indicators first.")
            return None
        
        try:
            # Run full analysis
            results = self.backtester.run_full_analysis(self.df_mftr)
            
            # Display probability analysis results
            prob_results = results['probability_analysis']
            print(f"\n📊 Probability Analysis Results:")
            print(f"  • Signals found: {prob_results['signals_found']}")
            print(f"  • Average return: {prob_results['avg_change']:.2f}%")
            print(f"  • Win rate: {prob_results['win_rate']:.2f}%")
            
            if 'statistics' in prob_results:
                stats = prob_results['statistics']
                print(f"  • Best trade: {stats['max_return']:.2f}%")
                print(f"  • Worst trade: {stats['min_return']:.2f}%")
                print(f"  • Volatility: {stats['std_return']:.2f}%")
            
            # Display optimization results
            opt_results = results['parameter_optimization']
            if opt_results['best_angle'] is not None:
                print(f"\n🎯 Parameter Optimization Results:")
                print(f"  • Best angle threshold: {opt_results['best_angle']}°")
                print(f"  • Best win rate: {opt_results['best_win_rate']:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ Error running analysis: {e}")
            return None
    
    def run_complete_analysis(self, symbol=None, timeframe=None, total_bars=None):
        """
        Run complete end-to-end analysis
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            total_bars (int): Number of bars to fetch
            
        Returns:
            dict: Complete analysis results
        """
        print("🚀 Starting MFTR Complete Analysis")
        print("=" * 50)
        
        # Step 1: Fetch data
        if not self.fetch_data(symbol, timeframe, total_bars):
            return None
        
        # Step 2: Calculate indicators
        if not self.calculate_indicators():
            return None
        
        # Step 3: Generate signals
        if not self.generate_signals():
            return None
        
        # Step 4: Run analysis
        results = self.run_analysis()
        
        if results:
            print("\n✅ Complete analysis finished successfully!")
            print("=" * 50)
        else:
            print("\n❌ Analysis failed")
        
        return results
    
    def get_system_status(self):
        """Get current system status"""
        return {
            'raw_data_loaded': self.df_raw is not None,
            'indicators_calculated': self.df_mftr is not None,
            'signals_generated': self.df_signals is not None,
            'data_points': len(self.df_mftr) if self.df_mftr is not None else 0
        }

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='MFTR Trading System Analysis')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', default='1h', help='Data timeframe')
    parser.add_argument('--bars', type=int, default=5000, help='Number of bars to fetch')
    parser.add_argument('--config-file', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize system
    config = Config()
    if args.config_file:
        # TODO: Implement config file loading
        print(f"Loading config from {args.config_file}")
    
    system = MFTRSystem(config)
    
    # Run complete analysis
    results = system.run_complete_analysis(
        symbol=args.symbol,
        timeframe=args.timeframe,
        total_bars=args.bars
    )
    
    if results:
        print("\n📋 Analysis Summary:")
        prob = results['probability_analysis']
        opt = results['parameter_optimization']
        
        print(f"Symbol: {args.symbol}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Data bars: {args.bars}")
        print(f"Valid signals: {prob['signals_found']}")
        print(f"Win rate: {prob['win_rate']:.2f}%")
        print(f"Average return: {prob['avg_change']:.2f}%")
        if opt['best_angle']:
            print(f"Optimal angle: {opt['best_angle']}°")
    
    return 0 if results else 1

if __name__ == "__main__":
    sys.exit(main())