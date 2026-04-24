import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import os

def get_mt5_data(symbol, timeframe, start_date, end_date):
    """
    Fetches historical data from MetaTrader 5 or loads from cache if available.
    
    Args:
        symbol (str): Trading symbol (e.g., 'EURUSD').
        timeframe (mt5.TIMEFRAME_*): Timeframe (e.g., mt5.TIMEFRAME_H1).
        start_date (datetime): Start date for data retrieval.
        end_date (datetime): End date for data retrieval.
        
    Returns:
        pd.DataFrame: Historical data with 'time', 'open', 'high', 'low', 'close', 'tick_volume'.
    """
    cache_dir = "data_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    cache_file = os.path.join(cache_dir, f"{symbol}_{timeframe}_{start_str}_{end_str}.pkl")
    
    if os.path.exists(cache_file):
        print(f"Cache hit! Loading data from {cache_file}")
        return pd.read_pickle(cache_file)

    # Connect to MetaTrader 5
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        return None

    # Fetch rates
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    # Shut down MT5 connection
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"No data found for {symbol} on the specified range.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Save to cache
    df.to_pickle(cache_file)
    print(f"Data saved to cache {cache_file}")
    
    return df

if __name__ == "__main__":
    # Test data retrieval
    timezone = pytz.timezone("Etc/UTC")
    start = datetime(2023, 1, 1, tzinfo=timezone)
    end = datetime(2024, 1, 1, tzinfo=timezone)
    
    # mt5.TIMEFRAME_H1 is 1 hour
    data = get_mt5_data("EURUSD", mt5.TIMEFRAME_H1, start, end)
    
    if data is not None:
        print(data.head())
        print(f"Total rows: {len(data)}")
