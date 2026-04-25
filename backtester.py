import pandas as pd
import numpy as np

def calculate_metrics(df, trades, prop_max_loss=-0.08, prop_target_profit=0.15):
    """
    Calculates backtesting metrics from the DataFrame and trades list.
    """
    if df.empty or not trades:
        return None

    # We use equity directly instead of compounding returns
    df['cumulative_returns'] = df['equity'] / df['equity'].iloc[0]
    
    total_return = df['cumulative_returns'].iloc[-1] - 1
    
    days = (df.index[-1] - df.index[0]).days
    if days > 0:
        annual_return = (1 + total_return) ** (365 / days) - 1
    else:
        annual_return = 0
        
    df['peak'] = df['cumulative_returns'].cummax()
    df['drawdown'] = (df['cumulative_returns'] - df['peak']) / df['peak']
    max_drawdown = df['drawdown'].min()
    
    # We will approximate standard metrics using MTM returns or trade returns
    daily_returns = df['strategy_returns']
    if daily_returns.std() != 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252 * (24*4)) # M15 scale
    else:
        sharpe_ratio = 0
        
    avg_holding_duration = 0 # Simplified
    
    pnl = total_return * 100 
    
    exit_trades = [t for t in trades if t['type'] == 'exit']
    total_exits = len(exit_trades)
    winning = len([t for t in exit_trades if t['pnl'] > 0])
    win_rate = winning / total_exits if total_exits > 0 else 0.0

    # Prop firm pass (Single linear track)
    cum_ret = df['cumulative_returns'].values
    running_peak = 1.0
    prop_firm_pass = False
    for val in cum_ret:
        if val > running_peak:
            running_peak = val
        current_dd = (val - running_peak) / running_peak
        current_ret = val - 1.0
        if current_dd <= -0.08:
            prop_firm_pass = False
            break
        if current_ret >= 0.15:
            prop_firm_pass = True
            break
            
    # Prop firm pass probability (Monte Carlo 1000 simulations with reset attempts)
    prop_firm_pass_prob = 0.0
    if total_exits >= 5:
        trade_pnls = np.array([t['pnl'] for t in exit_trades])
        n_sims = 1000
        total_mc_passes = 0
        total_mc_fails = 0
        
        for _ in range(n_sims):
            shuffled = np.random.permutation(trade_pnls)
            peak = 1.0
            val = 1.0
            
            for pnl_val in shuffled:
                val += pnl_val
                if val > peak:
                    peak = val
                
                dd = (val - peak) / peak
                ret = val - 1.0
                
                if dd <= prop_max_loss:
                    total_mc_fails += 1
                    # Reset account for next attempt
                    val = 1.0
                    peak = 1.0
                elif ret >= prop_target_profit:
                    total_mc_passes += 1
                    # Reset account for next attempt
                    val = 1.0
                    peak = 1.0
                    
        total_attempts = total_mc_passes + total_mc_fails
        if total_attempts > 0:
            prop_firm_pass_prob = total_mc_passes / total_attempts

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'prop_firm_pass': prop_firm_pass,
        'prop_firm_pass_prob': prop_firm_pass_prob,
        'avg_holding_duration_hours': avg_holding_duration,
        'pnl_percentage': pnl,
        'total_trades': total_exits
    }

def run_backtest(df, short_window=50, long_window=200, rr_ratio=1.0,
                 initial_balance=10000.0, risk_percent=0.01,
                 atr_period=14, atr_multiplier=1.5,
                 adx_threshold=20, adx_period=14,
                 sma_filter_period=200,
                 prop_max_loss=-0.08, prop_target_profit=0.15):
    """
    Runs the moving average crossover backtest with ATR dynamic SL/TP, ADX, and SMA directional filter.

    SL distance = ATR(atr_period) * atr_multiplier  (adapts to volatility)
    TP distance = SL distance * rr_ratio
    PnL is scaled so that hitting SL = exactly risk_percent loss of equity.

    Args:
        df              : OHLCV DataFrame from MT5.
        short_window    : Fast MA period.
        long_window     : Slow MA period.
        rr_ratio        : Reward / Risk multiplier (e.g. 2 = 1:2 R:R).
        initial_balance : Starting account balance in quote currency.
        risk_percent    : Fraction of equity to risk per trade (0.01 = 1%).
        atr_period      : Lookback period for ATR calculation.
        atr_multiplier  : Multiplier applied to ATR for SL distance.
        adx_threshold   : Minimum ADX value required to take a crossover trade (filters chop).
        adx_period      : Lookback period for ADX calculation.
        sma_filter_period : Long-term SMA period. If > 0, only take longs when price > SMA, shorts when < SMA.
    """
    df = df.copy()
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()

    # --- ATR calculation ---
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=atr_period).mean()

    # --- ADX calculation ---
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    tr_sum = df['tr'].rolling(window=adx_period).sum() + 1e-9 # avoid div zero
    plus_di = 100 * (df['plus_dm'].rolling(window=adx_period).sum() / tr_sum)
    minus_di = 100 * (df['minus_dm'].rolling(window=adx_period).sum() / tr_sum)
    
    df['dx'] = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    df['adx'] = df['dx'].rolling(window=adx_period).mean()

    # --- SMA Directional Filter ---
    if sma_filter_period > 0:
        df['sma_filter'] = df['close'].rolling(window=sma_filter_period).mean()
    else:
        df['sma_filter'] = np.nan

    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
    df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1

    # Crossover: only fires on the bar where signal changes AND ADX is above threshold AND aligned with SMA
    df['prev_signal'] = df['signal'].shift(1)
    df['crossover'] = 0
    
    long_cond = (df['signal'] == 1) & (df['prev_signal'] != 1) & (df['adx'] > adx_threshold)
    short_cond = (df['signal'] == -1) & (df['prev_signal'] != -1) & (df['adx'] > adx_threshold)
    
    if sma_filter_period > 0:
        long_cond = long_cond & (df['close'] > df['sma_filter'])
        short_cond = short_cond & (df['close'] < df['sma_filter'])
        
    df.loc[long_cond, 'crossover'] = 1   # bullish cross
    df.loc[short_cond, 'crossover'] = -1  # bearish cross
    
    df['strategy_returns'] = 0.0
    df['position'] = 0
    df['cumulative_returns'] = 1.0
    df['equity'] = initial_balance

    trades = []

    in_position = 0
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    current_equity = initial_balance
    current_sl_dist = 0.0   # SL distance for the current trade (used for PnL scaling)

    for i in range(1, len(df)):
        curr_idx = df.index[i]
        prev_idx = df.index[i-1]
        row = df.loc[curr_idx]
        prev_row = df.loc[prev_idx]
        
        pnl = 0.0
        
        # Check SL/TP if in position
        if in_position == 1:
            if row['low'] <= sl:
                pnl = -risk_percent
                in_position = 0
                current_equity += (pnl * initial_balance)
                trades.append({'time': str(curr_idx), 'type': 'exit', 'reason': 'sl', 'price': sl, 'pnl': pnl, 'equity': current_equity})
            elif row['high'] >= tp:
                pnl = risk_percent * rr_ratio
                in_position = 0
                current_equity += (pnl * initial_balance)
                trades.append({'time': str(curr_idx), 'type': 'exit', 'reason': 'tp', 'price': tp, 'pnl': pnl, 'equity': current_equity})
            elif row['signal'] == -1: # Reversal
                price_change = row['close'] - entry_price
                pnl = (price_change / current_sl_dist) * risk_percent if current_sl_dist != 0 else 0.0
                current_equity += (pnl * initial_balance)
                trades.append({'time': str(curr_idx), 'type': 'exit', 'reason': 'reversal', 'price': row['close'], 'pnl': pnl, 'equity': current_equity})
                # Enter Short
                atr_val = row['atr']
                if pd.notna(atr_val) and atr_val > 0:
                    current_sl_dist = atr_val * atr_multiplier
                    in_position = -1
                    entry_price = row['close']
                    sl = entry_price + current_sl_dist
                    tp = entry_price - current_sl_dist * rr_ratio
                    trades.append({'time': str(curr_idx), 'type': 'entry', 'direction': -1, 'price': entry_price, 'sl': sl, 'tp': tp, 'equity': current_equity})
            else:
                # MTM PnL
                price_change = row['close'] - prev_row['close']
                pnl = (price_change / current_sl_dist) * risk_percent if current_sl_dist != 0 else 0.0
                
        elif in_position == -1:
            if row['high'] >= sl:
                pnl = -risk_percent
                in_position = 0
                current_equity += (pnl * initial_balance)
                trades.append({'time': str(curr_idx), 'type': 'exit', 'reason': 'sl', 'price': sl, 'pnl': pnl, 'equity': current_equity})
            elif row['low'] <= tp:
                pnl = risk_percent * rr_ratio
                in_position = 0
                current_equity += (pnl * initial_balance)
                trades.append({'time': str(curr_idx), 'type': 'exit', 'reason': 'tp', 'price': tp, 'pnl': pnl, 'equity': current_equity})
            elif row['signal'] == 1: # Reversal
                price_change = entry_price - row['close']
                pnl = (price_change / current_sl_dist) * risk_percent if current_sl_dist != 0 else 0.0
                current_equity += (pnl * initial_balance)
                trades.append({'time': str(curr_idx), 'type': 'exit', 'reason': 'reversal', 'price': row['close'], 'pnl': pnl, 'equity': current_equity})
                # Enter Long
                atr_val = row['atr']
                if pd.notna(atr_val) and atr_val > 0:
                    current_sl_dist = atr_val * atr_multiplier
                    in_position = 1
                    entry_price = row['close']
                    sl = entry_price - current_sl_dist
                    tp = entry_price + current_sl_dist * rr_ratio
                    trades.append({'time': str(curr_idx), 'type': 'entry', 'direction': 1, 'price': entry_price, 'sl': sl, 'tp': tp, 'equity': current_equity})
            else:
                price_change = prev_row['close'] - row['close']
                pnl = (price_change / current_sl_dist) * risk_percent if current_sl_dist != 0 else 0.0
                
        else: # Not in position — only enter on a fresh crossover
            atr_val = row['atr']
            if pd.isna(atr_val) or atr_val <= 0:
                df.at[curr_idx, 'position'] = 0
                continue

            cross = row['crossover']
            if cross == 1:
                current_sl_dist = atr_val * atr_multiplier
                in_position = 1
                entry_price = row['close']
                sl = entry_price - current_sl_dist
                tp = entry_price + current_sl_dist * rr_ratio
                trades.append({'time': str(curr_idx), 'type': 'entry', 'direction': 1, 'price': entry_price, 'sl': sl, 'tp': tp, 'equity': current_equity})
            elif cross == -1:
                current_sl_dist = atr_val * atr_multiplier
                in_position = -1
                entry_price = row['close']
                sl = entry_price + current_sl_dist
                tp = entry_price - current_sl_dist * rr_ratio
                trades.append({'time': str(curr_idx), 'type': 'entry', 'direction': -1, 'price': entry_price, 'sl': sl, 'tp': tp, 'equity': current_equity})

        df.at[curr_idx, 'position'] = in_position
        df.at[curr_idx, 'strategy_returns'] = pnl
        df.at[curr_idx, 'equity'] = current_equity

    metrics = calculate_metrics(df, trades, prop_max_loss=prop_max_loss, prop_target_profit=prop_target_profit)
    
    return df, metrics, trades

if __name__ == "__main__":
    pass

