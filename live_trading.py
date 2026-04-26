import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import csv
import os
from datetime import datetime

# ======================================================================
#   CONFIGURATION
# ======================================================================
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
SHORT_WINDOW = 24
LONG_WINDOW = 52
RR_RATIO = 0.5
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
ADX_THRESHOLD = 20
ADX_PERIOD = 14
SMA_FILTER_PERIOD = 200
INITIAL_BALANCE = 10000.0  # Used for constant risk calculation
RISK_PERCENT = 0.01        # 1% risk of INITIAL_BALANCE per trade

CSV_LOG_FILE = "live_trades.csv"

# ======================================================================

def log_trade(action, symbol, lot_size, price, sl, tp, ticket):
    """
    Logs the executed trade into a CSV file.
    """
    file_exists = os.path.isfile(CSV_LOG_FILE)
    
    with open(CSV_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Action', 'Symbol', 'Lot Size', 'Price', 'SL', 'TP', 'Ticket'])
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, action, symbol, lot_size, price, sl, tp, ticket])
        
    print(f"[{timestamp}] Logged {action} trade for {symbol} to {CSV_LOG_FILE}")


def get_latest_signal_and_atr():
    """
    Retrieves historical data to calculate technical indicators and evaluate the latest signal from the *closed* candle.
    """
    # Fetch extra bars for accurate EMA/SMA and ADX calculations
    warmup_bars = max(LONG_WINDOW, SMA_FILTER_PERIOD) + ADX_PERIOD + 50
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, warmup_bars)
    
    if rates is None or len(rates) == 0:
        print(f"[{datetime.now()}] Failed to get rates from MT5")
        return None, None, None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate MAs
    df['short_ma'] = df['close'].rolling(window=SHORT_WINDOW).mean()
    df['long_ma'] = df['close'].rolling(window=LONG_WINDOW).mean()
    
    # Calculate ATR
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()
    
    # Calculate ADX
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    tr_sum = df['tr'].rolling(window=ADX_PERIOD).sum() + 1e-9
    plus_di = 100 * (df['plus_dm'].rolling(window=ADX_PERIOD).sum() / tr_sum)
    minus_di = 100 * (df['minus_dm'].rolling(window=ADX_PERIOD).sum() / tr_sum)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    df['adx'] = dx.rolling(window=ADX_PERIOD).mean()
    
    # Calculate SMA Filter
    df['sma_filter'] = df['close'].rolling(window=SMA_FILTER_PERIOD).mean()
    
    # Evaluate signal only based on the latest fully CLOSED candle (-2 index)
    # Index -1 represents the currently active, unclosed candle.
    row = df.iloc[-2]
    prev_row = df.iloc[-3]
    
    curr_short = row['short_ma']
    curr_long = row['long_ma']
    prev_short = prev_row['short_ma']
    prev_long = prev_row['long_ma']
    
    signal = 0
    
    # Check bullish crossover
    if curr_short > curr_long and prev_short <= prev_long:
        if row['adx'] > ADX_THRESHOLD and row['close'] > row['sma_filter']:
            signal = 1
            
    # Check bearish crossover
    elif curr_short < curr_long and prev_short >= prev_long:
        if row['adx'] > ADX_THRESHOLD and row['close'] < row['sma_filter']:
            signal = -1
            
    return signal, row['atr'], row['time']


def calculate_lot_size(sl_dist_price):
    """
    Calculates the lot size based on a fixed risk of the initial balance.
    """
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"Failed to get symbol info for {SYMBOL}")
        return None
        
    # Constant risk using initial balance
    dollar_risk = INITIAL_BALANCE * RISK_PERCENT
    
    contract_size = symbol_info.trade_contract_size
    lot = dollar_risk / (sl_dist_price * contract_size)
    
    # Normalize lot size to broker specifications
    lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
    lot = max(symbol_info.volume_min, min(symbol_info.volume_max, lot))
    
    return lot


def close_all_positions():
    """
    Closes all existing positions for the given symbol to prepare for a reversal.
    """
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None or len(positions) == 0:
        return
        
    for pos in positions:
        tick = mt5.symbol_info_tick(SYMBOL)
        action = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if action == mt5.ORDER_TYPE_SELL else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": action,
            "position": pos.ticket,
            "price": price,
            "magic": 999111,
            "comment": "MA Crossover Reversal",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close position {pos.ticket}, retcode: {res.retcode}")
        else:
            print(f"[{datetime.now()}] Closed position {pos.ticket} for reversal.")


def place_order(signal, atr_val):
    """
    Places a market order based on the signal with dynamic SL and TP.
    """
    action = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print(f"Failed to get tick for {SYMBOL}")
        return None
        
    price = tick.ask if signal == 1 else tick.bid
    
    # Distance in price units
    sl_dist = atr_val * ATR_MULTIPLIER
    
    lot_size = calculate_lot_size(sl_dist)
    if lot_size is None:
        return None
    
    if signal == 1:
        sl = price - sl_dist
        tp = price + sl_dist * RR_RATIO
    else:
        sl = price + sl_dist
        tp = price - sl_dist * RR_RATIO
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": action,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 999111,
        "comment": "MA Crossover Entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[{datetime.now()}] Order failed, retcode: {result.retcode}")
        print(result)
    else:
        print(f"[{datetime.now()}] Order placed successfully: Ticket {result.order}")
        log_trade("BUY" if signal == 1 else "SELL", SYMBOL, lot_size, price, sl, tp, result.order)
    
    return result


def main():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    print("=========================================")
    print(f"    Live Trading Bot Started    ")
    print("=========================================")
    print(f"Symbol: {SYMBOL} | Timeframe: M15 (Polled every minute)")
    print(f"Parameters: Short={SHORT_WINDOW}, Long={LONG_WINDOW}, ATR={ATR_PERIOD}(x{ATR_MULTIPLIER}), RR={RR_RATIO}, ADX > {ADX_THRESHOLD}")
    print(f"Risk Params: {RISK_PERCENT*100}% of ${INITIAL_BALANCE} (Constant Risk)")
    print(f"SMA Filter: {SMA_FILTER_PERIOD} periods")
    print("=========================================")
    print("Monitoring markets...")
    
    last_processed_candle_time = None
    
    try:
        while True:
            signal, atr_val, candle_time = get_latest_signal_and_atr()
            
            if signal is None:
                time.sleep(60)
                continue
                
            # If we haven't processed a signal for this closed candle timestamp
            if candle_time != last_processed_candle_time:
                last_processed_candle_time = candle_time
                
                # Check for active signal
                if signal != 0:
                    print(f"\n[{datetime.now()}] Valid Signal Generated: {'LONG' if signal == 1 else 'SHORT'}. Executing trade...")
                    
                    # Close previous position if exists for reversal
                    close_all_positions()
                    
                    # Place new position
                    place_order(signal, atr_val)
            
            # Wait 60 seconds before next check
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nBot strictly stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
