"""
Technical Indicators Module for MFTR Trading System
Contains all technical indicator calculations including custom MFTR
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from config import DEFAULT_MFTR_PARAMS

class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    def __init__(self, params=None):
        """Initialize with indicator parameters"""
        self.params = params or DEFAULT_MFTR_PARAMS
    
    def calculate_adx(self, df, length=14):
        """
        Self-contained ADX calculation to avoid pandas_ta DMI issues in Colab
        
        Args:
            df (pd.DataFrame): OHLCV data
            length (int): ADX calculation period
            
        Returns:
            pd.DataFrame: DataFrame with ADX values added
        """
        # Create a copy to avoid modifying original data
        df_copy = df.copy()
        
        # Calculate True Range and directional movements
        df_copy['tr'] = ta.true_range(df_copy['high'], df_copy['low'], df_copy['close'])
        df_copy['up_move'] = df_copy['high'].diff()
        df_copy['down_move'] = -df_copy['low'].diff()
        
        # Calculate Directional Movement
        df_copy['plus_dm'] = np.where(
            (df_copy['up_move'] > df_copy['down_move']) & (df_copy['up_move'] > 0), 
            df_copy['up_move'], 0
        )
        df_copy['minus_dm'] = np.where(
            (df_copy['down_move'] > df_copy['up_move']) & (df_copy['down_move'] > 0), 
            df_copy['down_move'], 0
        )
        
        # Calculate Directional Indicators
        df_copy['plus_di'] = 100 * (ta.ema(df_copy['plus_dm'], length) / ta.ema(df_copy['tr'], length))
        df_copy['minus_di'] = 100 * (ta.ema(df_copy['minus_dm'], length) / ta.ema(df_copy['tr'], length))
        
        # Calculate DX and ADX
        df_copy['dx'] = 100 * (abs(df_copy['plus_di'] - df_copy['minus_di']) / (df_copy['plus_di'] + df_copy['minus_di']))
        df_copy['adxValue'] = ta.ema(df_copy['dx'], length)
        
        # Clean up intermediate columns
        columns_to_drop = ['tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx']
        df_copy.drop(columns_to_drop, axis=1, inplace=True)
        
        return df_copy
    
    def calculate_base_indicators(self, df):
        """
        Calculate base indicators needed for MFTR
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with base indicators added
        """
        df_copy = df.copy()
        
        # RSI
        df_copy.ta.rsi(length=self.params['N_rsi'], append=True)
        
        # ATR
        df_copy.ta.atr(length=self.params['N_atr_ratio'], append=True)
        
        # ADX using custom calculation
        df_copy = self.calculate_adx(df_copy, length=self.params['dmi_adx_length'])
        
        # KAMA (Kaufman Adaptive Moving Average)
        df_copy['maValue'] = ta.kama(
            df_copy['close'], 
            length=self.params['kama_length'], 
            fast=self.params['kama_fast'], 
            slow=self.params['kama_slow']
        )
        
        return df_copy
    
    def calculate_mftr_components(self, df):
        """
        Calculate individual MFTR components
        
        Args:
            df (pd.DataFrame): DataFrame with base indicators
            
        Returns:
            pd.DataFrame: DataFrame with MFTR components added
        """
        df_copy = df.copy()
        
        # Price Ratio Component
        atr_column = f'ATRr_{self.params["N_atr_ratio"]}'
        df_copy['priceRatio'] = (df_copy['close'] - df_copy['maValue']) / df_copy[atr_column]
        
        # Volume-Weighted Close-Open Bias Component
        df_copy['vwcbRaw'] = (df_copy['close'] - df_copy['open']) * df_copy['volume']
        df_copy['vwcbSmoothed'] = ta.ema(df_copy['vwcbRaw'], length=self.params['N_vwcb_smooth'])
        
        # Cumulative Volume Delta Component
        df_copy['deltaVolume'] = np.where(
            df_copy['close'] > df_copy['open'], df_copy['volume'],
            np.where(df_copy['close'] < df_copy['open'], -df_copy['volume'], 0)
        )
        df_copy['cvd'] = ta.sma(df_copy['deltaVolume'], length=self.params['cvdLookback'])
        
        return df_copy
    
    def normalize_components(self, df):
        """
        Normalize MFTR components using z-score normalization
        
        Args:
            df (pd.DataFrame): DataFrame with MFTR components
            
        Returns:
            pd.DataFrame: DataFrame with normalized components
        """
        df_copy = df.copy()
        
        components_to_normalize = ['priceRatio', 'vwcbSmoothed', 'cvd']
        normalization_period = self.params['Normalization_Period']
        
        for comp in components_to_normalize:
            mean = df_copy[comp].rolling(window=normalization_period).mean()
            std = df_copy[comp].rolling(window=normalization_period).std()
            df_copy[f'norm_{comp}'] = (df_copy[comp] - mean) / std
        
        return df_copy
    
    def calculate_final_mftr(self, df):
        """
        Calculate final MFTR line and signal
        
        Args:
            df (pd.DataFrame): DataFrame with normalized components
            
        Returns:
            pd.DataFrame: DataFrame with final MFTR values
        """
        df_copy = df.copy()
        
        # Centered RSI
        rsi_column = f'RSI_{self.params["N_rsi"]}'
        df_copy['centered_rsi'] = df_copy[rsi_column] - 50
        
        # Raw MFTR Line
        scaling_factor = self.params['Scaling_Factor']
        df_copy['mftrLineRaw'] = (
            df_copy['norm_priceRatio'] * scaling_factor +
            df_copy['norm_vwcbSmoothed'] * scaling_factor +
            df_copy['norm_cvd'] * scaling_factor +
            df_copy['centered_rsi']
        )
        
        # Smoothed MFTR Line
        df_copy['mftrLine'] = ta.ema(df_copy['mftrLineRaw'], length=self.params['N_mftr_smooth'])
        
        # MFTR Signal Line
        df_copy['mftrSignal'] = ta.ema(df_copy['mftrLine'], length=self.params['N_signal'])
        
        # Calculate Angle (slope of MFTR line)
        delta_y = df_copy['mftrLine'] - df_copy['mftrLine'].shift(1)
        df_copy['angle'] = np.degrees(np.arctan(delta_y / 10))
        
        return df_copy
    
    def calculate_mftr(self, df):
        """
        Calculate complete MFTR indicator
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with all MFTR calculations, or None if error
        """
        if df is None or df.empty:
            print("Cannot calculate MFTR. DataFrame is empty.")
            return None
        
        print("Calculating MFTR indicator values...")
        
        try:
            # Step 1: Calculate base indicators
            df_with_base = self.calculate_base_indicators(df)
            
            # Step 2: Calculate MFTR components
            df_with_components = self.calculate_mftr_components(df_with_base)
            
            # Step 3: Normalize components
            df_normalized = self.normalize_components(df_with_components)
            
            # Step 4: Calculate final MFTR
            df_final = self.calculate_final_mftr(df_normalized)
            
            # Remove rows with NaN values
            df_final.dropna(inplace=True)
            
            print("MFTR calculation complete.")
            return df_final
            
        except Exception as e:
            print(f"A critical error occurred during MFTR calculation: {e}")
            return None
    
    def get_mftr_columns(self):
        """Return list of MFTR-related column names"""
        return [
            'adxValue', 'maValue', 'priceRatio', 'vwcbRaw', 'vwcbSmoothed',
            'deltaVolume', 'cvd', 'norm_priceRatio', 'norm_vwcbSmoothed', 
            'norm_cvd', 'centered_rsi', 'mftrLineRaw', 'mftrLine', 
            'mftrSignal', 'angle'
        ]

# Convenience functions for backward compatibility
def calculate_adx(df, length=14):
    """Convenience function for ADX calculation"""
    indicators = TechnicalIndicators()
    return indicators.calculate_adx(df, length)

def calculate_mftr(df, **params):
    """Convenience function for MFTR calculation"""
    indicators = TechnicalIndicators(params)
    return indicators.calculate_mftr(df)