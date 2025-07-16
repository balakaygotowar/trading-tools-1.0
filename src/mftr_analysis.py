# ==============================================================================
# MFTR Pro Analysis & Optimization Suite - Working Copy
# ==============================================================================
# Based on Sean's script (seans_script_7-10-25.py)
# This is our working version that can be modified safely
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
import pandas_ta as ta
import time

# ==============================================================================
# --- Data Loading Functions ---
# ==============================================================================
def load_data_from_csv(csv_filename='market_data.csv'):
    """
    Load historical OHLCV data from CSV file
    """
    import os
    from config import get_data_directory
    
    print(f"Loading data from CSV: {csv_filename}")
    try:
        data_dir = get_data_directory()
        csv_path = os.path.join(data_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        if df is not None and not df.empty:
            print(f"✅ Data loaded successfully! Total bars: {len(df)}")
            print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
            return df
        else:
            print("❌ CSV file is empty")
            return None
            
    except Exception as e:
        print(f"❌ Error loading CSV data: {e}")
        return None

def fetch_paginated_data(symbol='BTC/USDT', timeframe='1h', total_bars=5000, use_csv=True):
    """
    Load data from CSV by default, fallback to fetching from Coinbase API
    """
    if use_csv:
        df = load_data_from_csv()
        if df is not None:
            return df
        print("⚠️ CSV loading failed, falling back to API fetch...")
    
    print(f"Fetching {total_bars} bars of {symbol} data for the {timeframe} timeframe from Coinbase...")
    try:
        exchange = ccxt.coinbase()
        limit_per_request = 300
        all_ohlcv = []
        timeframe_duration_in_ms = exchange.parse_timeframe(timeframe) * 1000
        since = exchange.milliseconds() - total_bars * timeframe_duration_in_ms
        
        while len(all_ohlcv) < total_bars:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_request)
            if not ohlcv: 
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + timeframe_duration_in_ms
            print(f"Fetched {len(all_ohlcv)} bars so far...")
            time.sleep(exchange.rateLimit / 1000)
        
        df = pd.DataFrame(all_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.drop_duplicates(subset='time', inplace=True)
        df.set_index('time', inplace=True)
        print(f"\nData fetched successfully! Total bars collected: {len(df)}")
        return df.sort_index()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# ==============================================================================
# --- MFTR Indicator Calculation ---
# ==============================================================================
def calculate_adx(df, length=14):
    """
    Self-contained ADX calculation to avoid pandas_ta DMI issues in Colab
    """
    df['tr'] = ta.true_range(df['high'], df['low'], df['close'])
    df['up_move'] = df['high'].diff()
    df['down_move'] = -df['low'].diff()
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['plus_di'] = 100 * (ta.ema(df['plus_dm'], length) / ta.ema(df['tr'], length))
    df['minus_di'] = 100 * (ta.ema(df['minus_dm'], length) / ta.ema(df['tr'], length))
    df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
    df['adxValue'] = ta.ema(df['dx'], length)
    
    # Clean up intermediate columns
    df.drop(['tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx'], axis=1, inplace=True)
    return df

def calculate_mftr(df, **params):
    """
    Calculate the MFTR (Market Flow Trend Ratio) indicator
    """
    if df is None or df.empty:
        print("Cannot calculate MFTR. Dataframe is empty.")
        return None
    
    print("Calculating MFTR indicator values...")
    try:
        # Calculate base indicators
        df.ta.rsi(length=params['N_rsi'], append=True)
        df.ta.atr(length=params['N_atr_ratio'], append=True)
        
        # Calculate ADX using our custom function
        df = calculate_adx(df, length=params['dmi_adx_length'])
        
        # Calculate MFTR components
        df['maValue'] = ta.kama(df['close'], length=params['kama_length'], fast=params['kama_fast'], slow=params['kama_slow'])
        df['priceRatio'] = (df['close'] - df['maValue']) / df[f'ATRr_{params["N_atr_ratio"]}']
        df['vwcbRaw'] = (df['close'] - df['open']) * df['volume']
        df['vwcbSmoothed'] = ta.ema(df['vwcbRaw'], length=params['N_vwcb_smooth'])
        df['deltaVolume'] = np.where(df['close'] > df['open'], df['volume'], 
                                   np.where(df['close'] < df['open'], -df['volume'], 0))
        df['cvd'] = ta.sma(df['deltaVolume'], length=params['cvdLookback'])
        
        # Normalize components
        components_to_normalize = ['priceRatio', 'vwcbSmoothed', 'cvd']
        for comp in components_to_normalize:
            mean = df[comp].rolling(window=params['Normalization_Period']).mean()
            std = df[comp].rolling(window=params['Normalization_Period']).std()
            df[f'norm_{comp}'] = (df[comp] - mean) / std
        
        # Calculate final MFTR values
        df['centered_rsi'] = df[f'RSI_{params["N_rsi"]}'] - 50
        df['mftrLineRaw'] = (df['norm_priceRatio'] * params['Scaling_Factor'] + 
                           df['norm_vwcbSmoothed'] * params['Scaling_Factor'] + 
                           df['norm_cvd'] * params['Scaling_Factor'] + 
                           df['centered_rsi'])
        df['mftrLine'] = ta.ema(df['mftrLineRaw'], length=params['N_mftr_smooth'])
        df['mftrSignal'] = ta.ema(df['mftrLine'], length=params['N_signal'])
        
        # Calculate angle
        delta_y = df['mftrLine'] - df['mftrLine'].shift(1)
        df['angle'] = np.degrees(np.arctan(delta_y / 10))
        
        df.dropna(inplace=True)
        print("MFTR calculation complete.")
        return df
    except Exception as e:
        print(f"A critical error occurred during calculation: {e}")
        return None

# ==============================================================================
# --- Analysis Functions ---
# ==============================================================================
def analyze_event_probability(df, min_adx=20, min_angle_event=40, look_forward_bars=20):
    """
    Analyze probability of profitable trades based on signal criteria
    """
    if df is None: 
        return
    
    print(f"\n--- Running Probability Analysis ---")
    print(f"Event: Crossover with ADX > {min_adx} and Angle > {min_angle_event}")
    print(f"Analyzing price change over the next {look_forward_bars} bars.")
    
    # Find buy signals
    buy_signals = df[(df['mftrLine'] > df['mftrSignal']) & 
                    (df['mftrLine'].shift(1) < df['mftrSignal'].shift(1))]
    valid_buy_signals = buy_signals[(buy_signals['adxValue'] > min_adx) & 
                                   (buy_signals['angle'] > min_angle_event)]
    
    price_changes = []
    for index, row in valid_buy_signals.iterrows():
        start_index = df.index.get_loc(index)
        if start_index + look_forward_bars < len(df):
            entry_price = row['close']
            exit_price = df.iloc[start_index + look_forward_bars]['close']
            percent_change = ((exit_price - entry_price) / entry_price) * 100
            price_changes.append(percent_change)
    
    if not price_changes:
        print("No valid buy signals found for the specified criteria.")
    else:
        average_change = np.mean(price_changes)
        win_rate = (np.sum(np.array(price_changes) > 0) / len(price_changes)) * 100
        print(f"\nFound {len(valid_buy_signals)} valid buy signals.")
        print(f"Average price change after {look_forward_bars} bars: {average_change:.2f}%")
        print(f"Win Rate (price was higher after {look_forward_bars} bars): {win_rate:.2f}%")

def optimize_angle_parameter(df_original, min_adx=20, stop_loss_pct=2.0, take_profit_pct=4.0):
    """
    Optimize the minimum angle parameter using backtesting
    """
    if df_original is None: 
        return
    
    print(f"\n--- Running Parameter Optimization for 'minAngle' ---")
    angle_range = range(10, 51, 5)
    results = {}
    
    for test_angle in angle_range:
        df = df_original.copy()
        wins = 0
        losses = 0
        
        buy_signals = df[(df['mftrLine'] > df['mftrSignal']) & 
                        (df['mftrLine'].shift(1) < df['mftrSignal'].shift(1))]
        valid_buy_signals = buy_signals[(buy_signals['adxValue'] > min_adx) & 
                                       (buy_signals['angle'] > test_angle)]
        
        for index, row in valid_buy_signals.iterrows():
            entry_price = row['close']
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            take_profit_price = entry_price * (1 + take_profit_pct / 100)
            start_iloc = df.index.get_loc(index)
            
            for i in range(start_iloc + 1, len(df)):
                future_low = df.iloc[i]['low']
                future_high = df.iloc[i]['high']
                if future_low <= stop_loss_price:
                    losses += 1
                    break
                if future_high >= take_profit_price:
                    wins += 1
                    break
        
        total_trades = wins + losses
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        results[test_angle] = win_rate
        print(f"Testing Angle: {test_angle}, Found {total_trades} trades, Win Rate: {win_rate:.2f}%")
    
    if results:
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values(), width=2)
        plt.xlabel("Minimum Angle Setting")
        plt.ylabel("Win Rate (%)")
        plt.title("Optimization Results for Minimum Angle")
        plt.xticks(list(results.keys()))
        plt.grid(axis='y', linestyle='--')
        plt.show()

# ==============================================================================
# --- Main Execution ---
# ==============================================================================
def main():
    """
    Main function to run the complete analysis
    """
    # Default parameters
    default_params = {
        'kama_length': 10, 'kama_fast': 2, 'kama_slow': 30,
        'N_atr_ratio': 14, 'N_vwcb_smooth': 14,
        'cvdLookback': 21, 'N_rsi': 14, 'dmi_adx_length': 14,
        'Normalization_Period': 50, 'Scaling_Factor': 10.0,
        'N_mftr_smooth': 10, 'N_signal': 5
    }
    
    # Fetch data
    df = fetch_paginated_data(symbol='BTC/USDT', timeframe='1h', total_bars=5000)
    
    if df is not None:
        # Calculate MFTR
        df_mftr = calculate_mftr(df.copy(), **default_params)
        
        if df_mftr is not None:
            print(f"\nFinal dataframe is {len(df_mftr)} rows long. Head:")
            print(df_mftr.head())
            
            # Run analysis
            analyze_event_probability(df_mftr, min_adx=20, min_angle_event=40, look_forward_bars=24)
            optimize_angle_parameter(df_mftr, min_adx=20, stop_loss_pct=2.0, take_profit_pct=4.0)
        else:
            print("MFTR calculation failed.")
    else:
        print("Data fetching failed.")

if __name__ == "__main__":
    main()