"""
MFTR Trading System Package
A comprehensive trading analysis system centered around the MFTR indicator
"""

from .advanced_data_fetcher import AdvancedDataFetcher
from .advanced_indicators import AdvancedIndicators, DEFAULT_MFTR_PARAMS
from .probability_analyzer import ProbabilityAnalyzer
from .parameter_optimizer import ParameterOptimizer
from .mftr_system import MFTRSystem, run_mftr_analysis

__version__ = "1.0.0"
__author__ = "Trading Tools Team"

__all__ = [
    'AdvancedDataFetcher',
    'AdvancedIndicators', 
    'DEFAULT_MFTR_PARAMS',
    'ProbabilityAnalyzer',
    'ParameterOptimizer',
    'MFTRSystem',
    'run_mftr_analysis'
]