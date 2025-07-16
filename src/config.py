"""
Configuration management for MFTR Trading System
Contains all default parameters and settings
"""

import os

# Default MFTR Parameters
DEFAULT_MFTR_PARAMS = {
    'kama_length': 10,
    'kama_fast': 2, 
    'kama_slow': 30,
    'N_atr_ratio': 14,
    'N_vwcb_smooth': 14,
    'cvdLookback': 21,
    'N_rsi': 14,
    'dmi_adx_length': 14,
    'Normalization_Period': 50,
    'Scaling_Factor': 10.0,
    'N_mftr_smooth': 10,
    'N_signal': 5
}

# Data Fetching Configuration
DATA_CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframe': '1h',
    'total_bars': 5000,
    'limit_per_request': 300,
    'exchange': 'coinbase',
    'csv_filename': 'market_data.csv',
    'data_directory': 'data'  # Changed from 'price history' to 'data'
}

# Signal Generation Configuration  
SIGNAL_CONFIG = {
    'min_adx': 20,
    'min_angle_event': 40,
    'look_forward_bars': 24
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'stop_loss_pct': 2.0,
    'take_profit_pct': 4.0,
    'angle_range': range(10, 51, 5)
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'probability_analysis': {
        'min_adx': 20,
        'min_angle_event': 40,
        'look_forward_bars': 24
    },
    'parameter_optimization': {
        'min_adx': 20,
        'stop_loss_pct': 2.0,
        'take_profit_pct': 4.0
    }
}

class Config:
    """Configuration class for easy parameter management"""
    
    def __init__(self):
        self.mftr_params = DEFAULT_MFTR_PARAMS.copy()
        self.data_config = DATA_CONFIG.copy()
        self.signal_config = SIGNAL_CONFIG.copy()
        self.backtest_config = BACKTEST_CONFIG.copy()
        self.analysis_config = ANALYSIS_CONFIG.copy()
    
    def update_mftr_params(self, **kwargs):
        """Update MFTR parameters"""
        self.mftr_params.update(kwargs)
    
    def update_data_config(self, **kwargs):
        """Update data configuration"""
        self.data_config.update(kwargs)
    
    def update_signal_config(self, **kwargs):
        """Update signal configuration"""
        self.signal_config.update(kwargs)
    
    def get_all_config(self):
        """Return all configuration as dictionary"""
        return {
            'mftr_params': self.mftr_params,
            'data_config': self.data_config,
            'signal_config': self.signal_config,
            'backtest_config': self.backtest_config,
            'analysis_config': self.analysis_config
        }

def get_data_directory():
    """
    Get the absolute path to the data directory
    
    Returns:
        str: Absolute path to the data directory
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_name = DATA_CONFIG.get('data_directory', 'data')
    return os.path.join(base_dir, data_dir_name)