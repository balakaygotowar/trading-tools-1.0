# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## User commands
See folder ".claude/commands/" for .md files containing prompts from the user. 

## Environment Setup

### Python Environment
- Use Python 3.8+ for compatibility with common trading libraries
- Create virtual environment: `python -m venv venv`
- Activate virtual environment: `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`

### Google Colab Setup
This is also compatible with Google Colab. The main script contains pip install commands that should be run in Colab:

```bash
pip install --upgrade numpy==1.23.5 pandas ccxt pandas_ta
```

### Common Commands
- Install packages: `pip install <package_name>`
- Freeze dependencies: `pip freeze > requirements.txt`
- Run Python scripts: `python <script_name>.py`
- Run tests: `python -m pytest` (if using pytest)
- Format code: `python -m black .` (if using black)
- Lint code: `python -m flake8 .` (if using flake8)

## Project Structure

```
trading tools 1.0/
├── README.md (this file)
├── PROJECT_REQUIREMENTS.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── config.py
│   ├── data_fetcher.py
│   ├── indicators.py
│   ├── signal_generator.py
│   ├── backtester.py
│   └── mftr_analysis.py
├── data/
│   ├── market_data.csv
│   └── test_market_data.xlsx
├── tests/
│   └── test_data_fetcher.py
├── notebooks/
│   └── MFTR_Pro_Analysis_&_Optimization_Suite_(v5).ipynb
└── archive/
    ├── trading_tool_v0.5
    └── Trading Tools 1.0-20250716T032140Z-1-001.zip
```

This project is a comprehensive trading tools system with the following high-level architecture:

### Legacy Trading System Components
- **ConfigurationManager**: Handles loading, creating, updating, and validating configuration settings
- **GoogleAPIAuthenticator**: Manages authentication with Google APIs (Drive and Sheets)
- **DataGapChecker**: Identifies gaps in asset price history
- **DataRetriever**: Fetches and upserts missing price data
- **DataProcessor**: Performs ETL processes and calculates technical indicators
- **Backtester**: Runs backtests on active strategies with performance reporting
- **TradeManager**: Generates order signals for live trading
- **TradeExecutor**: Executes trades through brokers (Alpaca, Interactive Brokers)

### Strategy Processing Functions
- `process_live_strategies()`: Handles live trading execution
- `process_paper_trades()`: Manages paper trading simulation
- `process_backtesting()`: Runs backtesting on strategies

### Data Storage
- Early development: Price history stored as CSV files
- Asset price history stored in Google Drive as Sheets or CSV files
- Configuration file stores per-strategy settings

### Strategy Configuration
Each strategy supports multiple simultaneous statuses:
- **Backtesting status**: ENABLED, DISABLED
- **Paper trading status**: ENABLED, DISABLED  
- **Live trading status**: ENABLED, DISABLED

### Program Flow
1. **Initialization**: Load configurations, authenticate with Google APIs, retrieve folder IDs and top assets
2. **Data Processing**: Check for data gaps, fetch missing data, perform ETL and technical indicator calculations
3. **Strategy Execution**: Process strategies based on their active statuses (live/paper/backtesting)
4. **Trade Management**: Generate signals and execute trades for live strategies
5. **Reporting**: Generate backtest reports with profitability and performance metrics

## Core Architecture

### MFTR Trading System
The codebase implements a complete trading analysis system centered around the **MFTR (Market Flow Trend Ratio)** indicator:

**Data Pipeline:**
1. **Data Fetching** (`fetch_paginated_data`): Retrieves historical OHLCV data from Coinbase exchange using CCXT with pagination
2. **Indicator Calculation** (`calculate_mftr`): Computes the composite MFTR indicator 
3. **Signal Generation**: Identifies buy signals based on MFTR line crossovers with filtering criteria
4. **Analysis Functions**: Probability analysis and parameter optimization

**MFTR Indicator Components:**
- **Price Ratio**: (close - KAMA) / ATR - measures price deviation from adaptive moving average
- **VWCB (Volume-Weighted Close-Open Bias)**: (close - open) * volume - captures intrabar sentiment
- **CVD (Cumulative Volume Delta)**: Running sum of directional volume - tracks institutional flow
- **Centered RSI**: RSI - 50 - momentum component
- **Custom ADX**: Self-contained implementation to avoid library conflicts in Colab

All components are z-score normalized over a rolling window, then combined and smoothed.

### Key Functions

**Data Analysis:**
- `analyze_event_probability()`: Backtests signal performance over specified forward-looking periods
- `optimize_angle_parameter()`: Parameter sweep optimization testing different angle thresholds with stop-loss/take-profit logic

**Signal Filtering:**
- ADX > threshold (trend strength)
- Angle > threshold (momentum/slope)
- MFTR line crosses above signal line

## Running the Code

The script is designed to run top-to-bottom in Google Colab. Key execution points:

1. **Data Collection**: Fetches 5000 hourly BTC/USDT bars by default
2. **Indicator Calculation**: Applies MFTR with default parameters
3. **Analysis Execution**: Runs probability analysis and parameter optimization automatically

## Default Parameters

```python
default_params = {
    'kama_length': 10, 'kama_fast': 2, 'kama_slow': 30,
    'N_atr_ratio': 14, 'N_vwcb_smooth': 14,
    'cvdLookback': 21, 'N_rsi': 14, 'dmi_adx_length': 14,
    'Normalization_Period': 50, 'Scaling_Factor': 10.0,
    'N_mftr_smooth': 10, 'N_signal': 5
}
```

## Custom ADX Implementation

The codebase includes a self-contained ADX calculation (`calculate_adx`) to bypass pandas_ta DMI issues in Colab environments. This handles True Range, Directional Movement, and ADX smoothing internally.

## Analysis Configuration

- **Probability Analysis**: Tests 24-bar forward returns for signals with ADX > 20 and angle > 40
- **Parameter Optimization**: Tests angle thresholds from 10-50 with 2% stop-loss and 4% take-profit

## Development Guidelines
- Follow PEP 8 style guidelines for Python code
- Use type hints where appropriate for better code documentation
- Include docstrings for functions and classes
- Handle API keys and sensitive data through environment variables
- Use logging instead of print statements for production code