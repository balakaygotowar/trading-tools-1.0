# Project Requirements Documentation

## Overview
MFTR Pro Analysis & Optimization Suite - A comprehensive trading analysis system for cryptocurrency markets.

## Technical Requirements

### Python Dependencies
- **numpy==1.23.5** - Numerical computing (pinned for Colab compatibility)
- **pandas** - Data manipulation and analysis
- **ccxt** - Cryptocurrency exchange connectivity
- **pandas_ta** - Technical analysis indicators
- **matplotlib** - Data visualization and plotting

### Environment Requirements
- **Google Colab** - Primary execution environment
- **Python 3.x** - Runtime environment
- **Internet connectivity** - Required for live data fetching from exchanges

## Functional Requirements

### Data Requirements
- **Historical Data**: 5000+ bars of OHLCV data
- **Timeframe**: 1-hour candlesticks (configurable)
- **Symbol**: BTC/USDT (default, configurable)
- **Exchange**: Coinbase (via CCXT)
- **Data Output**: CSV file format for intermediate storage
- **File Management**: Overwrite same CSV file on each data fetch
- **Asset Tracking**: Include symbol column in CSV to identify asset ticker
- **File Structure**: All Python files in single directory for simplified imports

### Indicator Requirements
- **MFTR Calculation**: Custom composite indicator combining:
  - Price ratio (KAMA-based)
  - Volume-weighted close-bias
  - Cumulative volume delta
  - Centered RSI
  - Custom ADX implementation
- **Signal Generation**: Crossover detection with filtering
- **Normalization**: Z-score normalization over rolling windows

### Analysis Requirements
- **Probability Analysis**: Forward-looking performance testing
- **Parameter Optimization**: Systematic parameter sweeping
- **Risk Management**: Stop-loss and take-profit logic
- **Performance Metrics**: Win rate, average returns

## System Requirements

### Performance Requirements
- Handle 5000+ data points efficiently
- Paginated data fetching to avoid rate limits
- Real-time indicator calculations

### Data Flow Requirements
- **Data Fetcher Output**: Must save DataFrame to CSV file automatically
- **Downstream Functions**: Must read input data from staged CSV files only  
- **Data Pipeline**: CSV file serves as intermediate storage between components
- **CSV Format**: Include columns: time, open, high, low, close, volume, symbol
- **Import Simplification**: Flat directory structure eliminates complex import paths

### Compatibility Requirements
- **Colab Environment**: Must run in Google Colab notebooks
- **Library Conflicts**: Custom ADX to avoid pandas_ta DMI issues
- **Rate Limiting**: Respect exchange API limits

## Configuration Requirements

### Default Parameters
```python
{
    'kama_length': 10, 'kama_fast': 2, 'kama_slow': 30,
    'N_atr_ratio': 14, 'N_vwcb_smooth': 14,
    'cvdLookback': 21, 'N_rsi': 14, 'dmi_adx_length': 14,
    'Normalization_Period': 50, 'Scaling_Factor': 10.0,
    'N_mftr_smooth': 10, 'N_signal': 5
}
```

### Analysis Configuration
- **Minimum ADX**: 20 (trend strength filter)
- **Minimum Angle**: 40 (momentum filter)
- **Forward Bars**: 24 (probability analysis window)
- **Stop Loss**: 2%
- **Take Profit**: 4%

## Future Requirements (To Be Added)

### Planned Enhancements
- [ ] Multi-timeframe analysis
- [ ] Additional exchange support
- [ ] Real-time alerts
- [ ] Portfolio management features
- [ ] Advanced risk metrics

### Integration Requirements
- [ ] Database storage for historical results
- [ ] API endpoints for external access
- [ ] Automated execution capabilities
- [ ] Performance monitoring dashboard

---

**Note**: This document will be continuously updated as new requirements emerge. Please add new requirements below this line and periodically review for updates.