#!/usr/bin/env python3
"""
MFTR Analysis Example
Demonstrates how to use the converted MFTR notebook functionality
"""

import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mftr_system import MFTRSystem, run_mftr_analysis
from advanced_indicators import DEFAULT_MFTR_PARAMS


def main():
    """Run example MFTR analysis"""
    
    print("=== MFTR Trading System Example ===\n")
    
    # Option 1: Quick analysis using convenience function
    print("Running quick analysis...")
    results = run_mftr_analysis(
        symbol='BTC/USDT',
        timeframe='1h', 
        total_bars=2000  # Smaller dataset for example
    )
    
    print(f"\nQuick Analysis Results:")
    print(f"- Data points: {results.get('data_points', 0)}")
    print(f"- Total signals: {results['summary']['total_signals']}")
    print(f"- Win rate: {results['summary']['win_rate']:.1f}%")
    print(f"- Average return: {results['summary']['average_return']:.2f}%")
    
    # Option 2: Step-by-step analysis with custom parameters
    print("\n" + "="*50)
    print("Running detailed step-by-step analysis...")
    
    # Initialize system
    system = MFTRSystem(exchange='coinbase')
    
    # Fetch data
    print("\n1. Fetching data...")
    df = system.fetch_data('ETH/USDT', '4h', 1000)
    
    if df is not None:
        print(f"   Fetched {len(df)} bars of ETH/USDT 4h data")
        
        # Calculate indicators with custom parameters
        print("\n2. Calculating MFTR indicators...")
        custom_params = DEFAULT_MFTR_PARAMS.copy()
        custom_params['N_mftr_smooth'] = 8  # Faster MFTR smoothing
        custom_params['N_signal'] = 3       # Faster signal line
        
        df_mftr = system.calculate_indicators(custom_params)
        
        if df_mftr is not None:
            print(f"   Calculated indicators for {len(df_mftr)} data points")
            
            # Run probability analysis
            print("\n3. Running probability analysis...")
            prob_results = system.run_probability_analysis(
                min_adx=25,      # Higher ADX threshold
                min_angle=35,    # Lower angle threshold  
                look_forward_bars=12  # Shorter timeframe for 4h data
            )
            
            # Run parameter optimization
            print("\n4. Running parameter optimization...")
            opt_results = system.run_parameter_optimization(
                min_adx=25,
                stop_loss_pct=3.0,
                take_profit_pct=6.0,
                angle_range=range(20, 51, 5)
            )
            
            # Display detailed results
            print("\n5. Detailed Results:")
            print(f"   Probability Analysis:")
            print(f"   - Valid trades: {prob_results.get('valid_trades', 0)}")
            print(f"   - Win rate: {prob_results.get('win_rate', 0):.1f}%")
            print(f"   - Average return: {prob_results.get('average_return', 0):.2f}%")
            print(f"   - Best trade: {prob_results.get('best_trade', 0):.2f}%")
            print(f"   - Worst trade: {prob_results.get('worst_trade', 0):.2f}%")
            
            if opt_results:
                best_angle = max(opt_results, key=opt_results.get)
                print(f"\n   Parameter Optimization:")
                print(f"   - Best angle: {best_angle} ({opt_results[best_angle]:.1f}% win rate)")
                print(f"   - Angle range tested: {min(opt_results.keys())} to {max(opt_results.keys())}")
            
            # Generate visualization
            print("\n6. Generating charts...")
            system.plot_mftr_signals()
            
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()