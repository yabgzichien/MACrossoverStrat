# Moving Average Crossover Trading Bot Logic

This document provides a comprehensive overview of the logic and implementation of the Moving Average Crossover trading bot, including data retrieval, backtesting, and live trading components.

## 1. Data Retrieval (`mt5_data.py`)
This module handles the connection to MetaTrader 5 and fetches historical price data.

```python
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

def get_mt5_data(symbol, timeframe, start_date, end_date):
    """
    Fetches historical data from MetaTrader 5.
    """
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        return None

    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df
```

## 2. Backtesting Engine (`backtester.py`)
This module contains the core strategy logic and calculates performance metrics.

### Strategy Logic
- **Short MA**: Rolling average over a shorter window (e.g., 50).
- **Long MA**: Rolling average over a longer window (e.g., 200).
- **Signal**: 
  - `1` (Long) when Short MA > Long MA.
  - `-1` (Short) when Short MA < Long MA.

### Metrics Calculation
- **Total Return**: Cumulative return over the entire period.
- **Annual Return**: Returns scaled to a yearly basis.
- **Max Drawdown**: The largest peak-to-trough decline.
- **Sharpe Ratio**: Risk-adjusted returns.
- **Avg Holding Duration**: Average time spent in a trade.

```python
import pandas as pd
import numpy as np

def calculate_metrics(df):
    # ... (refer to code for full implementation)
    # Calculates PnL, Sharpe, Max Drawdown, etc.
    pass

def run_backtest(df, short_window=50, long_window=200):
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    
    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
    df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1
    
    df['position'] = df['signal']
    df['returns'] = df['close'].pct_change()
    
    metrics = calculate_metrics(df)
    return df, metrics
```

## 3. Main Execution and Visualization (`main.py`)
Orchestrates the backtesting process and visualizes the results.

```python
# ... (imports)
def main():
    # Configuration
    symbol = "EURUSD"
    # ...
    data = get_mt5_data(symbol, timeframe, start_date, end_date)
    results_df, metrics = run_backtest(data, short_window, long_window)
    # Print metrics and plot results using matplotlib
```

## 4. Live Trading Skeleton (`live_trading.py`)
A template for real-time execution. It monitors the market and places orders when signals change.

```python
# ... (imports)
def get_current_signal(symbol, timeframe, short_window, long_window):
    # Fetches recent data and calculates current MA crossover signal
    pass

def place_order(symbol, signal, lot_size):
    # Uses mt5.order_send to execute market orders
    pass

def main():
    # Infinite loop to monitor signals and execute trades
    while True:
        signal = get_current_signal(SYMBOL, TIMEFRAME, SHORT_WINDOW, LONG_WINDOW)
        if signal != current_position:
            place_order(SYMBOL, signal, LOT_SIZE)
        time.sleep(60)
```

## Metrics Definitions
- **PnL**: Percentage Profit and Loss.
- **Sharpe Ratio**: (Mean Daily Return / Std Dev of Daily Returns) * sqrt(252).
- **Max Drawdown**: (Current Value - Historical Peak) / Historical Peak.
- **Holding Duration**: Average time between trade entry and exit.
