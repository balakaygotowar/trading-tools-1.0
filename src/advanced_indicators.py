"""
Advanced Technical Indicators with Custom Implementations
Includes self-contained ADX calculation and complete MFTR implementation
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional


class AdvancedIndicators:
    """Advanced technical indicators with custom implementations"""
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """
        Self-contained ADX calculation to bypass pandas_ta issues
        
        Args:
            df: DataFrame with OHLC data
            length: Period for ADX calculation
            
        Returns:
            DataFrame with ADX values added
        """
        df = df.copy()
        
        # Calculate True Range and directional movement
        df['tr'] = ta.true_range(df['high'], df['low'], df['close'])
        df['up_move'] = df['high'].diff()
        df['down_move'] = -df['low'].diff()
        
        # Calculate +DM and -DM
        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0), 
            df['up_move'], 0
        )
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0), 
            df['down_move'], 0
        )
        
        # Calculate +DI and -DI
        df['plus_di'] = 100 * (ta.ema(df['plus_dm'], length) / ta.ema(df['tr'], length))
        df['minus_di'] = 100 * (ta.ema(df['minus_dm'], length) / ta.ema(df['tr'], length))
        
        # Calculate DX and ADX
        df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
        df['adxValue'] = ta.ema(df['dx'], length)
        
        # Clean up intermediate columns
        intermediate_cols = ['tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 
                           'plus_di', 'minus_di', 'dx']
        df.drop(intermediate_cols, axis=1, inplace=True, errors='ignore')
        
        return df
    
    @staticmethod
    def calculate_mftr(df: pd.DataFrame, **params) -> Optional[pd.DataFrame]:
        """
        Calculate complete MFTR indicator with all components
        
        Args:
            df: DataFrame with OHLCV data
            **params: MFTR parameters dictionary
            
        Returns:
            DataFrame with MFTR values or None if error
        """
        if df is None or df.empty:
            print("Cannot calculate MFTR. DataFrame is empty.")
            return None
            
        print("Calculating MFTR indicator values...")
        df = df.copy()
        
        try:
            # Calculate base indicators
            df.ta.rsi(length=params['N_rsi'], append=True)
            df.ta.atr(length=params['N_atr_ratio'], append=True)
            
            # Calculate custom ADX
            df = AdvancedIndicators.calculate_adx(df, length=params['dmi_adx_length'])
            
            # Calculate KAMA and price ratio
            df['maValue'] = ta.kama(
                df['close'], 
                length=params['kama_length'], 
                fast=params['kama_fast'], 
                slow=params['kama_slow']
            )
            df['priceRatio'] = (df['close'] - df['maValue']) / df[f'ATRr_{params["N_atr_ratio"]}']
            
            # Calculate VWCB (Volume-Weighted Close-Open Bias)
            df['vwcbRaw'] = (df['close'] - df['open']) * df['volume']
            df['vwcbSmoothed'] = ta.ema(df['vwcbRaw'], length=params['N_vwcb_smooth'])
            
            # Calculate CVD (Cumulative Volume Delta)
            df['deltaVolume'] = np.where(
                df['close'] > df['open'], df['volume'],
                np.where(df['close'] < df['open'], -df['volume'], 0)
            )
            df['cvd'] = ta.sma(df['deltaVolume'], length=params['cvdLookback'])
            
            # Normalize components using z-score
            components_to_normalize = ['priceRatio', 'vwcbSmoothed', 'cvd']
            for comp in components_to_normalize:
                mean = df[comp].rolling(window=params['Normalization_Period']).mean()
                std = df[comp].rolling(window=params['Normalization_Period']).std()
                df[f'norm_{comp}'] = (df[comp] - mean) / std
            
            # Calculate centered RSI
            df['centered_rsi'] = df[f'RSI_{params["N_rsi"]}'] - 50
            
            # Combine components into MFTR
            df['mftrLineRaw'] = (
                df['norm_priceRatio'] * params['Scaling_Factor'] +
                df['norm_vwcbSmoothed'] * params['Scaling_Factor'] +
                df['norm_cvd'] * params['Scaling_Factor'] +
                df['centered_rsi']
            )
            
            # Smooth MFTR line and create signal line
            df['mftrLine'] = ta.ema(df['mftrLineRaw'], length=params['N_mftr_smooth'])
            df['mftrSignal'] = ta.ema(df['mftrLine'], length=params['N_signal'])
            
            # Calculate angle (slope) of MFTR line
            delta_y = df['mftrLine'] - df['mftrLine'].shift(1)
            df['angle'] = np.degrees(np.arctan(delta_y / 10))
            
            # Clean up and return
            df.dropna(inplace=True)
            print("MFTR calculation complete.")
            return df
            
        except Exception as e:
            print(f"A critical error occurred during calculation: {e}")
            return None


# Default MFTR parameters
DEFAULT_MFTR_PARAMS = {
    'kama_length': 10, 'kama_fast': 2, 'kama_slow': 30,
    'N_atr_ratio': 14, 'N_vwcb_smooth': 14,
    'cvdLookback': 21, 'N_rsi': 14, 'dmi_adx_length': 14,
    'Normalization_Period': 50, 'Scaling_Factor': 10.0,
    'N_mftr_smooth': 10, 'N_signal': 5
}