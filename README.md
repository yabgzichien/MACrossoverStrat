# Moving Average Crossover Trading System

A robust algorithmic trading system for MetaTrader 5 (MT5), featuring a Walk-Forward Optimization (WFO) engine, Monte Carlo robustness testing, and a web-based visual replay engine.

## Project Structure

- `main.py`: The primary entry point for running a standard backtest.
- `optimizer.py`: Advanced optimization engine using Walk-Forward Optimization and Monte Carlo simulations.
- `backtester.py`: Core logic for strategy execution and performance metric calculation.
- `mt5_data.py`: Utility for fetching historical data from MT5 with chunking and caching.
- `live_trading.py`: Script for live execution on MT5 using the optimized strategy logic.
- `configs.txt`: Recommended parameters derived from optimization.
- `replay.html`: Self-contained interactive web dashboard for visual trade analysis.

## Core Strategy Features

- **Moving Average Crossover**: Uses Short and Long EMA for trend signals.
- **Dynamic Risk Management**: Stop-Loss (SL) and Take-Profit (TP) levels are calculated dynamically using ATR (Average True Range).
- **Trend Filter**: Includes a long-term SMA (e.g., 200 SMA) filter to ensure trades align with major trend direction.
- **Volatility Filter**: ADX (Average Directional Index) requirement to avoid trading in choppy/sideways markets.
- **Fixed Risk Sizing**: Every trade risks a fixed percentage of the initial balance (standard) or current equity (optional compounding).

## Setup & Requirements

1.  **MetaTrader 5**: Ensure the MT5 terminal is installed and logged into an account.
2.  **Dependencies**: Install required Python libraries:
    ```bash
    pip install MetaTrader5 pandas numpy pytz
    ```
3.  **MT5 Connection**: The MT5 terminal must be running while executing scripts that fetch data (`main.py`, `optimizer.py`).

## Usage

### 1. Running a Standard Backtest
Modify the parameters in `main.py` (Symbol, Timeframe, Date range, MA windows, etc.) and run:
```bash
python main.py
```
This will:
- Fetch or load cached historical data.
- Run the backtest and print metrics (Sharpe Ratio, Win Rate, Drawdown, etc.).
- Generate `backtest_data.csv` and `backtest_trades.csv`.
- Create `replay.html` for visual analysis.

### 2. Strategy Optimization (WFO)
The optimizer uses Walk-Forward Optimization to find the most robust parameters over various market conditions, reducing the risk of overfitting.
```bash
python optimizer.py
```
Key Features:
- **Random Search**: Efficiently samples parameter combinations.
- **Prop Firm Scoring**: Optimizes for the highest probability of passing a prop firm challenge (profit target vs. drawdown limit).
- **Monte Carlo Shuffle**: Shuffles trade order to test the robustness of the strategy against sequence risk.
Results are saved to `optimization_results.csv` and summarized in the terminal.

### 3. Visual Replay Engine
Open `replay.html` in any web browser to:
- View the price chart with technical indicators.
- Replay trades one by one or at adjustable speeds.
- Analyze entry/exit points, SL/TP levels, and real-time equity curve.

### 4. Live Trading
Once you have found optimal parameters, update the constants in `live_trading.py` and run:
```bash
python live_trading.py
```
The bot polls the market every minute, evaluates signals on the **last closed candle**, and logs all trades to `live_trades.csv`.

## Data Management
- Historical data is cached in the `data_cache/` directory to prevent redundant MT5 downloads.
- To refresh data, simply delete the relevant file in the cache directory or clear the folder.

## Disclaimer
Algorithmic trading involves significant risk. This project is for educational and research purposes. Always test on a demo account before risking real capital.
