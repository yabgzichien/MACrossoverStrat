import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime

# Configuration
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
SHORT_WINDOW = 24
LONG_WINDOW = 52
RISK_PIPS = 100  # Distance in "points" (e.g. 100 = $1.00 move for Gold if point=0.01)
RR_RATIO = 1.0   # Risk-Reward Ratio
RISK_PERCENT = 0.01 # 1% risk per trade


def get_current_signal(symbol, timeframe, short_window, long_window):
    """
    Retrieves historical data and determines the current strategy signal.
    """
    # Fetch enough bars to calculate the long MA
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, long_window + 50)
    
    if rates is None or len(rates) == 0:
        return 0 # Neutral if no data

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate MAs
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    
    # Get last signal
    last_row = df.iloc[-1]
    
    if last_row['short_ma'] > last_row['long_ma']:
        return 1 # Long
    elif last_row['short_ma'] < last_row['long_ma']:
        return -1 # Short
    else:
        return 0 # Neutral

def calculate_lot_size(symbol, risk_percent, sl_pips):
    """
    Calculates the lot size based on account equity and risk pips.
    """
    account_info = mt5.account_info()
    if account_info is None:
        return None
    
    equity = account_info.equity
    dollar_risk = equity * risk_percent
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None
        
    point = symbol_info.point
    sl_dist = sl_pips * point
    
    # Lot size calculation: lot = dollar_risk / (sl_dist * tick_value / tick_size)
    # Simplified: Assuming 1 lot = 100,000 units (Forex) or standard for Gold
    # For Gold, 1 lot = 100oz. So $1 move = $100 per 1 lot.
    # dollar_risk / (sl_dist * standard_contract_size)
    
    contract_size = symbol_info.trade_contract_size
    lot = dollar_risk / (sl_dist * contract_size)
    
    # Normalize lot size
    lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
    lot = max(symbol_info.volume_min, min(symbol_info.volume_max, lot))
    
    return lot

def place_order(symbol, signal, sl_pips, tp_pips, risk_percent):
    """
    Places a market order based on the signal with SL and TP.
    """
    # Check current position
    # Simplified: This function just places an order.
    # In a real bot, you'd check if a position is already open and close it first.
    
    action = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick for {symbol}")
        return None
        
    price = tick.ask if signal == 1 else tick.bid
    
    point_info = mt5.symbol_info(symbol)
    if point_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None
        
    point = point_info.point
    
    lot_size = calculate_lot_size(symbol, risk_percent, sl_pips)
    if lot_size is None:
        print(f"Failed to calculate lot size for {symbol}")
        return None
    
    # Distance in price units
    sl_dist = sl_pips * point
    tp_dist = tp_pips * point
    
    if signal == 1:
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": action,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 123456,
        "comment": "MA Crossover Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, retcode: {result.retcode}")
    else:
        print(f"Order placed successfully: {result.order}")
    
    return result

def main():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    print(f"Bot started. Monitoring {SYMBOL} on {TIMEFRAME}...")
    
    current_position = 0 # 0: None, 1: Long, -1: Short
    
    try:
        while True:
            # 1. Get current signal
            signal = get_current_signal(SYMBOL, TIMEFRAME, SHORT_WINDOW, LONG_WINDOW)
            
            # 2. Check for signal change
            if signal != current_position:
                print(f"Signal changed to {signal}. Executing trade...")
                
                # Close previous position if exists (Simplified)
                # In real code, you would use mt5.positions_get to find and close orders.
                
                # Place new order
                if signal != 0:
                    tp_pips = RISK_PIPS * RR_RATIO
                    place_order(SYMBOL, signal, RISK_PIPS, tp_pips, RISK_PERCENT)
                    current_position = signal

            
            # 3. Wait for the next check (e.g., every minute)
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("Bot stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
