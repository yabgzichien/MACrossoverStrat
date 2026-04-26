import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas as pd

if not mt5.initialize():
    print('Failed to initialize')
    exit()

timezone = pytz.timezone('Etc/UTC')
start_date = datetime(2020, 2, 1, tzinfo=timezone)
end_date = datetime(2026, 3, 1, tzinfo=timezone)
symbol = 'XAUUSD'
timeframe = mt5.TIMEFRAME_M15

chunk_size = 50000
all_rates = []
current_end = end_date
start_compare = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date

loops = 0
while loops < 15: # limit chunks
    rates = mt5.copy_rates_from(symbol, timeframe, current_end, chunk_size)
    if rates is None or len(rates) == 0:
        print(f'Rates returned empty on loop {loops}')
        break
    df_chunk = pd.DataFrame(rates)
    df_chunk['time'] = pd.to_datetime(df_chunk['time'], unit='s')
    all_rates.append(df_chunk)
    
    min_time = df_chunk['time'].min()
    min_compare = min_time.replace(tzinfo=None) if min_time.tzinfo else min_time
    print(f'Chunk {loops}: from {min_time} to {df_chunk["time"].max()}, bars: {len(rates)}')
    
    if min_compare <= start_compare:
        print('Reached start_date')
        break
    if len(rates) < chunk_size:
        break
        
    next_end = min_time.to_pydatetime()
    if current_end.tzinfo:
        next_end = next_end.replace(tzinfo=current_end.tzinfo)
    
    if next_end >= current_end:
        print('next_end >= current_end, breaking built-in infinite loop')
        break
    current_end = next_end
    loops += 1

mt5.shutdown()

if all_rates:
    df = pd.concat(all_rates)
    df.drop_duplicates(subset=['time'], inplace=True)
    df.sort_values(by='time', inplace=True)
    df = df[(df['time'] >= start_compare)]
    print(f"Total rows fetched: {len(df)}")
    print(df.head(2))
    print(df.tail(2))
