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

    # Fetch rates in chunks to bypass MT5 download limits
    chunk_size = 50000
    all_rates = []
    current_end = end_date
    start_compare = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date

    while True:
        rates = mt5.copy_rates_from(symbol, timeframe, current_end, chunk_size)
        if rates is None or len(rates) == 0:
            break
            
        df_chunk = pd.DataFrame(rates)
        df_chunk['time'] = pd.to_datetime(df_chunk['time'], unit='s')
        all_rates.append(df_chunk)
        
        min_time = df_chunk['time'].min()
        min_compare = min_time.replace(tzinfo=None) if min_time.tzinfo else min_time
        
        if min_compare <= start_compare:
            break
        if len(rates) < chunk_size:
            break
            
        next_end = min_time.to_pydatetime()
        if current_end.tzinfo:
            next_end = next_end.replace(tzinfo=current_end.tzinfo)
            
        if next_end >= current_end:
            break
        current_end = next_end
        
    # Shut down MT5 connection
    mt5.shutdown()

    if not all_rates:
        print(f"No data found for {symbol} on the specified range.")
        return None

    # Concatenate all chunks and clean up
    df = pd.concat(all_rates)
    df.drop_duplicates(subset=['time'], inplace=True)
    df.sort_values(by='time', inplace=True)
    
    # Filter strictly to the requested range
    end_compare = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
    df = df[(df['time'] >= start_compare) & (df['time'] <= end_compare)]
    
    if df.empty:
        print(f"No data found for {symbol} on the specified range after filtering.")
        return None
        
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
