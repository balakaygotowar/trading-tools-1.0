"""
MFTR Trading System Package
A modular cryptocurrency trading analysis system based on the MFTR indicator
"""

__version__ = "1.0.0"
__author__ = "Trading Team"
__description__ = "MFTR Pro Analysis & Optimization Suite"

# Import main classes for easy access
from .config import Config, DEFAULT_MFTR_PARAMS, DATA_CONFIG, SIGNAL_CONFIG, BACKTEST_CONFIG
from .data_fetcher.data_fetcher import DataFetcher, fetch_paginated_data
from .indicators import TechnicalIndicators, calculate_adx, calculate_mftr
from .signal_generator import SignalGenerator, generate_buy_signals, get_signal_details
from .backtester import Backtester, analyze_event_probability, optimize_angle_parameter
from .main import MFTRSystem

# Define what gets imported with "from mftr_system import *"
__all__ = [
    # Main system
    'MFTRSystem',
    
    # Core classes
    'Config',
    'DataFetcher',
    'TechnicalIndicators', 
    'SignalGenerator',
    'Backtester',
    
    # Configuration
    'DEFAULT_MFTR_PARAMS',
    'DATA_CONFIG',
    'SIGNAL_CONFIG',
    'BACKTEST_CONFIG',
    
    # Convenience functions
    'fetch_paginated_data',
    'calculate_adx',
    'calculate_mftr',
    'generate_buy_signals',
    'get_signal_details',
    'analyze_event_probability',
    'optimize_angle_parameter',
]

def get_version():
    """Return package version"""
    return __version__

def get_info():
    """Return package information"""
    return {
        'name': 'MFTR Trading System',
        'version': __version__,
        'author': __author__,
        'description': __description__
    }