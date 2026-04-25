"""
Walk-Forward Optimization (WFO) + Monte Carlo Robustness Engine
================================================================
Tunes: short_window, long_window, rr_ratio, atr_period, atr_multiplier
Method: Random Search (fast, avoids curse of dimensionality)
Assessment: win_rate, sharpe_ratio, max_drawdown, prop_firm_pass
Anti-overfit: WFO rolling windows + Monte Carlo trade-shuffle
"""

import sys
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from backtester import run_backtest

# =================== CONFIGURATION ===================

# Parameter ranges for random sampling
PARAM_RANGES = {
    'short_window':   [10, 15, 20, 24, 30, 40],
    'long_window':    [40, 52, 60, 80, 100],
    'rr_ratio':       [1, 1.25, 1.5, 1.75],
    'atr_period':     [10, 14, 20, 30],
    'atr_multiplier': [0.5 ,1.0, 1.5, 2.0, 2.5],
}

# Number of random samples per WFO fold
N_RANDOM_SAMPLES = 100

# WFO settings
WFO_IN_SAMPLE_MONTHS  = 6
WFO_OUT_SAMPLE_MONTHS = 2
WFO_STEP_MONTHS       = 2

# Monte Carlo settings
MC_SIMULATIONS = 100

# Prop firm challenge thresholds
PROP_PROFIT_TARGET = 0.15   # 15% profit target
PROP_MAX_DD_LIMIT  = -0.08  # 8% max drawdown limit

# Scoring weights
W_SHARPE   = 0.2
W_WINRATE  = 0
W_DRAWDOWN = 0.5
W_PROPFIRM = 0.3   # Reward params that pass the prop firm test

# Backtest constants
INITIAL_BALANCE = 10000.0
RISK_PERCENT    = 0.01


def log(msg=""):
    """Print with immediate flush for real-time output."""
    print(msg)
    sys.stdout.flush()


# =================== SCORING FUNCTION ===================

def composite_score(metrics):
    """
    Composite score from win_rate, sharpe, max_drawdown, and prop_firm_pass.
    Higher is better.
    """
    if metrics is None or metrics.get('total_trades', 0) < 5:
        return -999

    sharpe  = metrics['sharpe_ratio']
    winrate = metrics['win_rate']
    dd      = 1 + metrics['max_drawdown']  # -0.20 -> 0.80
    prop    = 1.0 if metrics.get('prop_firm_pass', False) else 0.0

    return (W_SHARPE * sharpe) + (W_WINRATE * winrate) + (W_DRAWDOWN * dd) + (W_PROPFIRM * prop)


# =================== RANDOM SEARCH ===================

def sample_random_params(n=N_RANDOM_SAMPLES):
    """Generate N random parameter combinations (short < long enforced)."""
    combos = []
    attempts = 0
    while len(combos) < n and attempts < n * 10:
        p = {k: random.choice(v) for k, v in PARAM_RANGES.items()}
        if p['short_window'] < p['long_window']:
            combos.append(p)
        attempts += 1
    return combos


# =================== WALK-FORWARD ENGINE ===================

def split_wfo_folds(df):
    """Generate sliding walk-forward folds."""
    start = df.index[0]
    end = df.index[-1]
    folds = []

    is_end_offset = pd.DateOffset(months=WFO_IN_SAMPLE_MONTHS)
    oos_duration  = pd.DateOffset(months=WFO_OUT_SAMPLE_MONTHS)
    step          = pd.DateOffset(months=WFO_STEP_MONTHS)

    fold_start = start
    while True:
        is_end  = fold_start + is_end_offset
        oos_end = is_end + oos_duration

        if oos_end > end:
            break

        is_df  = df[fold_start:is_end]
        oos_df = df[is_end:oos_end]

        if len(is_df) > 100 and len(oos_df) > 20:
            folds.append((is_df, oos_df))

        fold_start = fold_start + step

    return folds


def optimize_on_fold(is_df, combos, fold_num):
    """Random search on in-sample data. Returns (best_params, best_score)."""
    best_score = -999
    best_params = None
    total = len(combos)
    t0 = time.time()

    for idx, params in enumerate(combos):
        try:
            _, metrics, _ = run_backtest(
                is_df,
                short_window=params['short_window'],
                long_window=params['long_window'],
                rr_ratio=params['rr_ratio'],
                initial_balance=INITIAL_BALANCE,
                risk_percent=RISK_PERCENT,
                atr_period=params['atr_period'],
                atr_multiplier=params['atr_multiplier'],
            )
            score = composite_score(metrics)
        except Exception:
            score = -999

        if score > best_score:
            best_score = score
            best_params = params

        # Progress every 20%
        if (idx + 1) % max(1, total // 5) == 0:
            elapsed = time.time() - t0
            pct = (idx + 1) / total * 100
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            log(f"    Fold {fold_num}: {pct:.0f}% ({idx+1}/{total}) "
                f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s | best={best_score:.4f}")

    return best_params, best_score


def evaluate_oos(oos_df, params):
    """Run best IS params on out-of-sample data."""
    _, metrics, trades = run_backtest(
        oos_df,
        short_window=params['short_window'],
        long_window=params['long_window'],
        rr_ratio=params['rr_ratio'],
        initial_balance=INITIAL_BALANCE,
        risk_percent=RISK_PERCENT,
        atr_period=params['atr_period'],
        atr_multiplier=params['atr_multiplier'],
    )
    return metrics, trades


def run_wfo(df):
    """Full Walk-Forward Optimization with random search per fold."""
    folds = split_wfo_folds(df)

    if not folds:
        log("ERROR: Not enough data for WFO folds. Extend the date range.")
        return [], []

    log(f"\n{'='*70}")
    log(f"  WALK-FORWARD OPTIMIZATION")
    log(f"  {len(folds)} folds | {N_RANDOM_SAMPLES} random samples per fold")
    log(f"{'='*70}")

    fold_results = []
    all_oos_trades = []
    total_t0 = time.time()

    for i, (is_df, oos_df) in enumerate(folds):
        log(f"\n--- Fold {i+1}/{len(folds)} ---")
        log(f"  IS:  {is_df.index[0].date()} -> {is_df.index[-1].date()}  ({len(is_df)} bars)")
        log(f"  OOS: {oos_df.index[0].date()} -> {oos_df.index[-1].date()}  ({len(oos_df)} bars)")

        # Generate fresh random combos for each fold
        combos = sample_random_params(N_RANDOM_SAMPLES)

        # 1. Optimize on IS
        fold_t0 = time.time()
        best_params, is_score = optimize_on_fold(is_df, combos, i + 1)
        fold_time = time.time() - fold_t0

        log(f"  Best IS params: {best_params}")
        log(f"  IS score: {is_score:.4f}  (took {fold_time:.1f}s)")

        # 2. Validate on OOS
        oos_metrics, oos_trades = evaluate_oos(oos_df, best_params)
        oos_score = composite_score(oos_metrics) if oos_metrics else -999

        if oos_metrics:
            prop_str = "PASS" if oos_metrics.get('prop_firm_pass', False) else "FAIL"
            log(f"  OOS -> Return: {oos_metrics['total_return']*100:.2f}%  "
                f"Sharpe: {oos_metrics['sharpe_ratio']:.2f}  "
                f"WinRate: {oos_metrics['win_rate']*100:.1f}%  "
                f"MaxDD: {oos_metrics['max_drawdown']*100:.2f}%  "
                f"PropFirm: {prop_str}  "
                f"Trades: {oos_metrics['total_trades']}")
        else:
            log(f"  OOS -> No trades generated")

        fold_results.append({
            'fold': i + 1,
            'is_start': str(is_df.index[0].date()),
            'is_end': str(is_df.index[-1].date()),
            'oos_start': str(oos_df.index[0].date()),
            'oos_end': str(oos_df.index[-1].date()),
            'is_score': is_score,
            'oos_score': oos_score,
            **{f'param_{k}': v for k, v in best_params.items()},
            **(
                {f'oos_{k}': v for k, v in oos_metrics.items()}
                if oos_metrics else {}
            ),
        })

        # Collect OOS trade PnLs for Monte Carlo
        exit_pnls = [t['pnl'] for t in oos_trades if t['type'] == 'exit']
        all_oos_trades.extend(exit_pnls)

    total_time = time.time() - total_t0
    log(f"\nWFO completed in {total_time:.1f}s")

    return fold_results, all_oos_trades


# =================== MONTE CARLO ENGINE ===================

def prop_firm_sim(equity_curve, target=PROP_PROFIT_TARGET, dd_limit=PROP_MAX_DD_LIMIT):
    """Check if an equity curve passes the prop firm challenge."""
    peak = 1.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (val - peak) / peak
        ret = val - 1.0
        if dd <= dd_limit:
            return False
        if ret >= target:
            return True
    return False


def run_monte_carlo(trade_pnls, n_sims=MC_SIMULATIONS):
    """
    Shuffle trade order N times, replay equity curve each time.
    Includes prop firm pass probability.
    """
    if len(trade_pnls) < 5:
        log("\nMonte Carlo: Too few OOS trades for meaningful simulation.")
        return None

    trade_pnls = np.array(trade_pnls)
    n_trades = len(trade_pnls)

    log(f"\n{'='*70}")
    log(f"  MONTE CARLO SIMULATION  ({n_sims} runs, {n_trades} OOS trades)")
    log(f"{'='*70}")

    results = []
    prop_passes = 0

    for sim in range(n_sims):
        shuffled = np.random.permutation(trade_pnls)

        # Replay equity curve
        equity = np.ones(n_trades + 1)
        for j, pnl in enumerate(shuffled):
            equity[j + 1] = equity[j] * (1 + pnl)

        final_return = equity[-1] - 1
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        max_dd = drawdowns.min()

        # Sharpe from trade-level returns
        if shuffled.std() != 0:
            sharpe = (shuffled.mean() / shuffled.std()) * np.sqrt(n_trades)
        else:
            sharpe = 0

        wins = np.sum(shuffled > 0)
        win_rate = wins / n_trades

        # Prop firm check
        passed = prop_firm_sim(equity)
        if passed:
            prop_passes += 1

        results.append({
            'final_return': final_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'prop_firm_pass': passed,
        })

        if (sim + 1) % 200 == 0:
            log(f"  MC progress: {sim+1}/{n_sims}")

    mc_df = pd.DataFrame(results)

    # Print percentile summary
    log(f"\n{'Metric':<20} {'P5':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P95':>10}")
    log("-" * 70)
    for col in ['final_return', 'max_drawdown', 'sharpe_ratio', 'win_rate']:
        p5  = mc_df[col].quantile(0.05)
        p25 = mc_df[col].quantile(0.25)
        p50 = mc_df[col].quantile(0.50)
        p75 = mc_df[col].quantile(0.75)
        p95 = mc_df[col].quantile(0.95)

        if col in ('final_return', 'max_drawdown', 'win_rate'):
            log(f"{col:<20} {p5*100:>9.2f}% {p25*100:>9.2f}% {p50*100:>9.2f}% {p75*100:>9.2f}% {p95*100:>9.2f}%")
        else:
            log(f"{col:<20} {p5:>10.3f} {p25:>10.3f} {p50:>10.3f} {p75:>10.3f} {p95:>10.3f}")

    # Prop firm pass probability
    prop_rate = prop_passes / n_sims * 100
    log(f"\n{'='*70}")
    log(f"  PROP FIRM PASS PROBABILITY: {prop_rate:.1f}%")
    log(f"  ({prop_passes}/{n_sims} simulations hit +{PROP_PROFIT_TARGET*100:.0f}% before -{abs(PROP_MAX_DD_LIMIT)*100:.0f}% DD)")
    log(f"{'='*70}")

    # Robustness check
    p5_return = mc_df['final_return'].quantile(0.05)
    if p5_return > 0:
        log(f"\n[ROBUST] 95% of shuffled simulations are profitable (P5 return = {p5_return*100:.2f}%)")
    else:
        log(f"\n[NOT ROBUST] P5 return = {p5_return*100:.2f}% (need > 0% for 95% confidence)")

    return mc_df


# =================== MAIN ===================

def main():
    import MetaTrader5 as mt5
    import pytz
    from mt5_data import get_mt5_data

    # --- Configuration ---
    symbol    = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M15
    timezone  = pytz.timezone("Etc/UTC")
    start_date = datetime(2023, 1, 1, tzinfo=timezone)
    end_date   = datetime(2024, 1, 1, tzinfo=timezone)

    log(f"Loading data for {symbol} ({start_date.date()} -> {end_date.date()})...")
    data = get_mt5_data(symbol, timeframe, start_date, end_date)

    if data is None:
        log("Failed to fetch data. Ensure MT5 is open.")
        return

    log(f"Loaded {len(data)} bars.\n")

    # --- Walk-Forward Optimization ---
    fold_results, all_oos_trades = run_wfo(data)

    if not fold_results:
        return

    # Save WFO results
    wfo_df = pd.DataFrame(fold_results)
    wfo_df.to_csv("optimization_results.csv", index=False)
    log(f"\nSaved WFO results to 'optimization_results.csv'")

    # --- WFO Summary ---
    log(f"\n{'='*70}")
    log(f"  WFO SUMMARY")
    log(f"{'='*70}")

    param_cols = [c for c in wfo_df.columns if c.startswith('param_')]
    param_counts = wfo_df[param_cols].apply(tuple, axis=1).value_counts()
    most_common_params = dict(zip(
        [c.replace('param_', '') for c in param_cols],
        param_counts.index[0]
    ))

    log(f"\nMost frequently selected parameters across folds:")
    for k, v in most_common_params.items():
        log(f"  {k}: {v}")

    avg_oos_return = wfo_df['oos_total_return'].mean() if 'oos_total_return' in wfo_df else 0
    avg_oos_sharpe = wfo_df['oos_sharpe_ratio'].mean() if 'oos_sharpe_ratio' in wfo_df else 0
    avg_oos_wr     = wfo_df['oos_win_rate'].mean() if 'oos_win_rate' in wfo_df else 0
    avg_oos_dd     = wfo_df['oos_max_drawdown'].mean() if 'oos_max_drawdown' in wfo_df else 0
    oos_prop_passes = sum(1 for _, r in wfo_df.iterrows() if r.get('oos_prop_firm_pass', False))

    log(f"\nAggregated OOS Performance:")
    log(f"  Avg Return:       {avg_oos_return*100:.2f}%")
    log(f"  Avg Sharpe:       {avg_oos_sharpe:.3f}")
    log(f"  Avg Win Rate:     {avg_oos_wr*100:.1f}%")
    log(f"  Avg Max DD:       {avg_oos_dd*100:.2f}%")
    log(f"  Prop Firm Passes: {oos_prop_passes}/{len(wfo_df)} folds")

    # --- Monte Carlo ---
    mc_df = run_monte_carlo(all_oos_trades)

    if mc_df is not None:
        mc_df.to_csv("monte_carlo_results.csv", index=False)
        log(f"\nSaved Monte Carlo results to 'monte_carlo_results.csv'")

    # --- Final recommendation ---
    log(f"\n{'='*70}")
    log(f"  RECOMMENDED PARAMETERS")
    log(f"{'='*70}")
    for k, v in most_common_params.items():
        log(f"  {k} = {v}")
    log(f"\nCopy these into main.py to use the optimized strategy.")


if __name__ == "__main__":
    main()
