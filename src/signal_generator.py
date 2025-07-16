"""
Signal Generation Module for MFTR Trading System
Handles detection and validation of trading signals
"""

import pandas as pd
import numpy as np
from config import SIGNAL_CONFIG

class SignalGenerator:
    """Class for generating trading signals from MFTR data"""
    
    def __init__(self, config=None):
        """Initialize with signal configuration"""
        self.config = config or SIGNAL_CONFIG
    
    def detect_crossovers(self, df):
        """
        Detect MFTR line crossovers above signal line
        
        Args:
            df (pd.DataFrame): DataFrame with MFTR data
            
        Returns:
            pd.DataFrame: DataFrame with buy signals marked
        """
        df_copy = df.copy()
        
        # Detect crossovers where MFTR line crosses above signal line
        current_above = df_copy['mftrLine'] > df_copy['mftrSignal']
        previous_below = df_copy['mftrLine'].shift(1) < df_copy['mftrSignal'].shift(1)
        
        df_copy['crossover_signal'] = current_above & previous_below
        
        return df_copy
    
    def apply_adx_filter(self, df, min_adx=None):
        """
        Filter signals based on ADX strength
        
        Args:
            df (pd.DataFrame): DataFrame with crossover signals
            min_adx (float): Minimum ADX value for valid signals
            
        Returns:
            pd.DataFrame: DataFrame with ADX-filtered signals
        """
        min_adx = min_adx or self.config['min_adx']
        df_copy = df.copy()
        
        df_copy['adx_filter'] = df_copy['adxValue'] > min_adx
        
        return df_copy
    
    def apply_angle_filter(self, df, min_angle=None):
        """
        Filter signals based on MFTR line angle (momentum)
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            min_angle (float): Minimum angle for valid signals
            
        Returns:
            pd.DataFrame: DataFrame with angle-filtered signals
        """
        min_angle = min_angle or self.config['min_angle_event']
        df_copy = df.copy()
        
        df_copy['angle_filter'] = df_copy['angle'] > min_angle
        
        return df_copy
    
    def generate_buy_signals(self, df, min_adx=None, min_angle=None):
        """
        Generate complete buy signals with all filters applied
        
        Args:
            df (pd.DataFrame): DataFrame with MFTR data
            min_adx (float): Minimum ADX threshold
            min_angle (float): Minimum angle threshold
            
        Returns:
            pd.DataFrame: DataFrame with validated buy signals
        """
        min_adx = min_adx or self.config['min_adx']
        min_angle = min_angle or self.config['min_angle_event']
        
        # Step 1: Detect crossovers
        df_with_crossovers = self.detect_crossovers(df)
        
        # Step 2: Apply ADX filter
        df_with_adx = self.apply_adx_filter(df_with_crossovers, min_adx)
        
        # Step 3: Apply angle filter
        df_with_angle = self.apply_angle_filter(df_with_adx, min_angle)
        
        # Step 4: Combine all filters for final buy signal
        df_with_angle['buy_signal'] = (
            df_with_angle['crossover_signal'] & 
            df_with_angle['adx_filter'] & 
            df_with_angle['angle_filter']
        )
        
        return df_with_angle
    
    def get_signal_details(self, df):
        """
        Get detailed information about generated signals
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            
        Returns:
            dict: Signal statistics and details
        """
        if 'buy_signal' not in df.columns:
            return {'error': 'No buy signals found in DataFrame'}
        
        # Count different types of signals
        total_crossovers = df['crossover_signal'].sum() if 'crossover_signal' in df.columns else 0
        adx_filtered = df['adx_filter'].sum() if 'adx_filter' in df.columns else 0
        angle_filtered = df['angle_filter'].sum() if 'angle_filter' in df.columns else 0
        final_signals = df['buy_signal'].sum()
        
        # Get signal locations
        signal_dates = df[df['buy_signal']].index.tolist()
        
        # Calculate filtering efficiency
        crossover_to_final_ratio = (final_signals / total_crossovers * 100) if total_crossovers > 0 else 0
        
        return {
            'total_crossovers': total_crossovers,
            'adx_passed': adx_filtered,
            'angle_passed': angle_filtered,
            'final_buy_signals': final_signals,
            'filtering_efficiency': f"{crossover_to_final_ratio:.2f}%",
            'signal_dates': signal_dates,
            'signal_frequency': f"{final_signals / len(df) * 100:.2f}% of all bars"
        }
    
    def get_signals_dataframe(self, df):
        """
        Get DataFrame containing only the signal rows
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            
        Returns:
            pd.DataFrame: DataFrame with only signal rows
        """
        if 'buy_signal' not in df.columns:
            return pd.DataFrame()
        
        return df[df['buy_signal']].copy()
    
    def validate_signals(self, df):
        """
        Validate signal quality and provide warnings
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            
        Returns:
            dict: Validation results and warnings
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        required_columns = ['mftrLine', 'mftrSignal', 'adxValue', 'angle']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation['valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_columns}")
            return validation
        
        # Check for sufficient data
        if len(df) < 50:
            validation['warnings'].append("Limited data may reduce signal reliability")
        
        # Check ADX values
        avg_adx = df['adxValue'].mean()
        if avg_adx < 15:
            validation['warnings'].append(f"Low average ADX ({avg_adx:.2f}) may indicate weak trends")
        
        # Check for signal frequency
        if 'buy_signal' in df.columns:
            signal_count = df['buy_signal'].sum()
            signal_frequency = signal_count / len(df)
            
            if signal_frequency > 0.1:  # More than 10% of bars have signals
                validation['warnings'].append("High signal frequency may indicate over-sensitivity")
            elif signal_frequency < 0.01:  # Less than 1% of bars have signals
                validation['warnings'].append("Low signal frequency may indicate under-sensitivity")
        
        return validation

# Convenience functions
def generate_buy_signals(df, min_adx=20, min_angle=40):
    """Convenience function for generating buy signals"""
    generator = SignalGenerator()
    return generator.generate_buy_signals(df, min_adx, min_angle)

def get_signal_details(df):
    """Convenience function for getting signal details"""
    generator = SignalGenerator()
    return generator.get_signal_details(df)