import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import json
from mt5_data import get_mt5_data
from backtester import run_backtest

def main():
    # Configuration
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M15
    short_window = 24
    long_window = 52
    rr_ratio = 1.5       # Reward / Risk multiplier (1 = 1:1, 2 = 1:2, etc.)
    initial_balance = 10000.0  # Starting account balance (in quote currency)
    risk_percent = 0.01 # Risk 1% of equity per trade
    atr_period = 14       # ATR lookback period (bars)
    atr_multiplier = 1  # SL distance = ATR * this multiplier
    adx_threshold = 20    # Minimum ADX value to take a trade (filters chop)
    adx_period = 14       # ADX lookback period
    sma_filter_period = 200 # SMA directional filter (0 to disable)
    prop_max_loss = -0.1   # Max drawdown limit for prop firm pass probability
    prop_target_profit = 0.08 # Profit target for prop firm pass probability
    use_compounding = False # Set to True to allow lot sizing to grow/shrink with equity
    crossover_exec_bars = 0 # Execute trade N bars after the crossover event

    # Define date range
    timezone = pytz.timezone("Etc/UTC")
    start_date = datetime(2023, 2, 1, tzinfo=timezone)
    end_date = datetime(2024, 3, 1, tzinfo=timezone)
    
    print(f"--- Starting Backtest for {symbol} ---")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    # 1. Fetch data
    data = get_mt5_data(symbol, timeframe, start_date, end_date)
    
    if data is None:
        print("Failed to fetch data. Ensure MetaTrader 5 is open and connected.")
        return

    # 2. Run backtest
    results_df, metrics, trades = run_backtest(
        data, short_window, long_window,
        rr_ratio=rr_ratio,
        initial_balance=initial_balance,
        risk_percent=risk_percent,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        adx_threshold=adx_threshold,
        adx_period=adx_period,
        sma_filter_period=sma_filter_period,
        prop_max_loss=prop_max_loss,
        prop_target_profit=prop_target_profit,
        use_compounding=use_compounding,
        crossover_exec_bars=crossover_exec_bars
    )

    
    # 3. Output metrics and save to CSV
    if metrics:
        print("\n--- Backtesting Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print(f"\nInitial Balance: ${initial_balance:,.2f}")
        print(f"Final Balance:   ${initial_balance * (1 + metrics['total_return']):,.2f}")

        print("\nSaving results to CSV...")
        results_df.to_csv("backtest_data.csv")
        
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv("backtest_trades.csv", index=False)
            print("Saved 'backtest_data.csv' and 'backtest_trades.csv'")

    
    # 4. Build self-contained replay HTML (data embedded inline to avoid CORS issues)
    print("\nExporting data for Visual Replay Engine...")
    chart_data = []
    for idx, row in results_df.iterrows():
        chart_data.append({
            'time': int(idx.timestamp()),
            'open': round(float(row['open']), 5),
            'high': round(float(row['high']), 5),
            'low': round(float(row['low']), 5),
            'close': round(float(row['close']), 5),
            'short_ma': None if pd.isna(row['short_ma']) else round(float(row['short_ma']), 5),
            'long_ma': None if pd.isna(row['long_ma']) else round(float(row['long_ma']), 5),
            'sma_filter': None if ('sma_filter' not in row or pd.isna(row['sma_filter'])) else round(float(row['sma_filter']), 5),
            'equity': round(float(row.get('cumulative_returns', 1.0)) * initial_balance, 2)
        })

    processed_trades = []
    for t in trades:
        tc = dict(t)
        tc['time'] = int(pd.to_datetime(tc['time']).timestamp())
        processed_trades.append(tc)

    dump_data = {
        'symbol': symbol,
        'metrics': {k: (round(v, 6) if isinstance(v, float) else v) for k, v in (metrics or {}).items()},
        'chart_data': chart_data,
        'trades': processed_trades,
        'initial_balance': initial_balance,
        'long_window': long_window
    }

    json_str = json.dumps(dump_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Replay Engine – {symbol}</title>
    <meta charset="utf-8">
    <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: #131722; color: #d1d4dc; font-family: -apple-system, system-ui, sans-serif; padding: 20px; }}
        h2 {{ margin-bottom: 14px; font-size: 18px; color: #fff; }}
        .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }}
        button {{ padding: 8px 16px; cursor: pointer; background: #2962ff; color: #fff; border: none; border-radius: 4px; font-weight: 600; font-size: 13px; transition: background .15s; }}
        button:hover {{ background: #1e53e5; }}
        label {{ font-size: 13px; color: #787b86; }}
        input[type=number] {{ width: 60px; padding: 6px; background: #1e222d; border: 1px solid #363a45; border-radius: 4px; color: #d1d4dc; font-size: 13px; }}
        .metrics {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 14px; background: #1e222d; padding: 14px 18px; border-radius: 6px; }}
        .metric-card {{ display: flex; flex-direction: column; gap: 4px; }}
        .metric-title {{ color: #787b86; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }}
        .metric-value {{ font-weight: 700; font-size: 17px; }}
        .win {{ color: #4CAF50; }}
        .loss {{ color: #FF5252; }}
        #chart {{ width: 100%; height: 580px; border-radius: 6px; overflow: hidden; }}
        .progress {{ font-size: 12px; color: #787b86; margin-left: auto; }}
    </style>
</head>
<body>
    <h2>Strategy Replay Engine – <span id="symLabel"></span></h2>
    <div class="metrics" id="metricsDisplay"></div>
    <!-- Trade info panel: shows active SL/TP as text -->
    <div id="tradePanel" style="display:none; background:#1e222d; border-radius:6px;
         padding:10px 18px; margin-bottom:12px; font-size:13px; display:flex;
         gap:22px; align-items:center; flex-wrap:wrap;"></div>
    <div class="controls">
        <button id="playBtn">▶ Play</button>
        <button id="pauseBtn">⏸ Pause</button>
        <button id="stepBtn">⏭ Step</button>
        <button id="resetBtn">⟳ Reset</button>
        <label>Speed (ms/bar):</label>
        <input type="number" id="speed" value="30" min="1" max="2000">
        <span class="progress" id="progressLabel">Bar 0 / 0</span>
    </div>
    <div id="chart"></div>

<script>
    const replayData = {json_str};

    document.getElementById('symLabel').textContent = replayData.symbol;

    const m = replayData.metrics;
    const ib = replayData.initial_balance || 10000;
    
    // We will build dynamic DOM elements for the real-time stats
    document.getElementById('metricsDisplay').innerHTML = `
        <div class="metric-card">
            <span class="metric-title">Live Equity</span>
            <span class="metric-value win" id="rtEquity">$${{ib.toFixed(2)}}</span>
        </div>
        <div class="metric-card">
            <span class="metric-title">Live Return</span>
            <span class="metric-value win" id="rtReturn">0.00%</span>
        </div>
        <div class="metric-card">
            <span class="metric-title" style="color:#2962ff">Short MA (${{replayData.short_window || 24}})</span>
        </div>
        <div class="metric-card">
            <span class="metric-title" style="color:#ff9800">Long MA (${{replayData.long_window || 52}})</span>
        </div>
        <div class="metric-card">
            <span class="metric-title" style="color:#B2B5BE">200 SMA Filter</span>
        </div>
        <div style="margin-left:auto; text-align:right" class="metric-card">
            <span class="metric-title">Final Stats</span>
            <span class="metric-value" style="font-size:13px">WinRate: ${{(m.win_rate*100).toFixed(1)}}% | Sharpe: ${{m.sharpe_ratio.toFixed(2)}}</span>
        </div>
    `;

    const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
        width: document.getElementById('chart').clientWidth,
        height: 580,
        layout: {{ backgroundColor: '#131722', textColor: '#d1d4dc' }},
        grid: {{ vertLines: {{ color: '#2B2B43' }}, horzLines: {{ color: '#2B2B43' }} }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
        rightPriceScale: {{ borderColor: '#485c7b' }},
        timeScale: {{ borderColor: '#485c7b', timeVisible: true, secondsVisible: false }}
    }});

    new ResizeObserver(() => chart.resize(document.getElementById('chart').clientWidth, 580)).observe(document.getElementById('chart'));

    const candleSeries = chart.addCandlestickSeries({{
        upColor: '#26a69a', downColor: '#ef5350',
        borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350'
    }});
    const shortMaSeries = chart.addLineSeries({{ color: '#2962ff', lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
    const longMaSeries  = chart.addLineSeries({{ color: '#ff9800', lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
    const smaFilterSeries = chart.addLineSeries({{ color: '#B2B5BE', lineWidth: 2, lineStyle: 1, priceLineVisible: false, lastValueVisible: false }});

    // KEY FIX: extend auto-scale to always include SL/TP levels
    // createPriceLine is excluded from LWC auto-scale by default!
    let activeTrade = null;
    candleSeries.applyOptions({{
        autoscaleInfoProvider: (baseImpl) => {{
            const base = baseImpl();
            if (!activeTrade) return base;
            const lo = Math.min(activeTrade.sl, activeTrade.tp);
            const hi = Math.max(activeTrade.sl, activeTrade.tp);
            if (!base) return {{ priceRange: {{ minValue: lo, maxValue: hi }}, margins: {{ above: 0.1, below: 0.1 }} }};
            return {{
                priceRange: {{
                    minValue: Math.min(base.priceRange.minValue, lo),
                    maxValue: Math.max(base.priceRange.maxValue, hi)
                }},
                margins: base.margins || {{ above: 0.1, below: 0.1 }}
            }};
        }}
    }});

    let currentIndex = 0;
    let isPlaying = false;
    let timer = null;
    let activeMarkers = [];
    const total = replayData.chart_data.length;
    const PREFILL = replayData.long_window || 50;  // warm-up = long MA window

    // Active price line handles
    let slPriceLine = null;
    let tpPriceLine = null;
    let entryPriceLine = null;

    // Build a trade lookup map keyed by bar timestamp for O(1) access
    const tradeMap = {{}};
    for (const t of replayData.trades) {{
        if (!tradeMap[t.time]) tradeMap[t.time] = [];
        tradeMap[t.time].push(t);
    }}

    function updateTradePanel(direction, entry, sl, tp) {{
        const p = document.getElementById('tradePanel');
        p.style.display = 'flex';
        p.innerHTML = `
            <span style="color:#787b86">Position:</span>
            <span style="font-weight:700; color:${{direction===1?'#2962ff':'#ff9800'}}"
            >${{direction===1?'▲ Long':'▼ Short'}}</span>
            <span style="color:#787b86">Entry:</span>
            <span style="font-weight:700">${{entry.toFixed(5)}}</span>
            <span style="color:#FF5252">&#9632; SL:</span>
            <span style="color:#FF5252; font-weight:700">${{sl.toFixed(5)}}</span>
            <span style="color:#4CAF50">&#9632; TP:</span>
            <span style="color:#4CAF50; font-weight:700">${{tp.toFixed(5)}}</span>
        `;
    }}
    function hideTradePanel() {{
        const p = document.getElementById('tradePanel');
        p.style.display = 'none';
        p.innerHTML = '';
    }}

    function updateProgress() {{
        document.getElementById('progressLabel').textContent = `Bar ${{currentIndex}} / ${{total}}`;
    }}

    function clearPriceLines() {{
        if (slPriceLine) {{ candleSeries.removePriceLine(slPriceLine); slPriceLine = null; }}
        if (tpPriceLine) {{ candleSeries.removePriceLine(tpPriceLine); tpPriceLine = null; }}
        if (entryPriceLine) {{ candleSeries.removePriceLine(entryPriceLine); entryPriceLine = null; }}
    }}

    function drawPriceLines(sl, tp, direction, entryPrice) {{
        clearPriceLines();
        entryPriceLine = candleSeries.createPriceLine({{
            price: entryPrice,
            color: direction === 1 ? '#2962ff' : '#ff9800',
            lineWidth: 2,
            lineStyle: 1,
            axisLabelVisible: true,
            title: 'ENTRY'
        }});
        slPriceLine = candleSeries.createPriceLine({{
            price: sl,
            color: '#FF5252',
            lineWidth: 2,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'SL'
        }});
        tpPriceLine = candleSeries.createPriceLine({{
            price: tp,
            color: '#4CAF50',
            lineWidth: 2,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'TP'
        }});
        if (direction !== undefined) updateTradePanel(direction, entryPrice, sl, tp);
    }}

    // Restore SL/TP lines at current active trade state (called after warmup ends)
    function restoreActivePriceLines() {{
        if (activeTrade) {{
            drawPriceLines(activeTrade.sl, activeTrade.tp, activeTrade.direction, activeTrade.price);
        }} else {{
            clearPriceLines();
            hideTradePanel();
        }}
    }}

    function updateChart() {{
        if (currentIndex >= total) {{ pausePlayback(); return; }}
        const bar = replayData.chart_data[currentIndex];
        candleSeries.update({{ time: bar.time, open: bar.open, high: bar.high, low: bar.low, close: bar.close }});
        if (bar.short_ma !== null) shortMaSeries.update({{ time: bar.time, value: bar.short_ma }});
        if (bar.long_ma  !== null) longMaSeries.update({{ time: bar.time, value: bar.long_ma }});
        if (bar.sma_filter !== null) smaFilterSeries.update({{ time: bar.time, value: bar.sma_filter }});

        if (bar.equity !== undefined) {{
            const rtReturn = ((bar.equity - ib) / ib * 100).toFixed(2);
            document.getElementById('rtEquity').textContent = `$${{bar.equity.toFixed(2)}}`;
            document.getElementById('rtReturn').textContent = `${{rtReturn}}%`;
            document.getElementById('rtReturn').className = `metric-value ${{rtReturn >= 0 ? 'win' : 'loss'}}`;
        }}

        const localTrades = tradeMap[bar.time] || [];
        let markersChanged = false;
        for (const t of localTrades) {{
            if (t.type === 'entry') {{
                activeTrade = {{ sl: t.sl, tp: t.tp, direction: t.direction, price: t.price }};
                drawPriceLines(t.sl, t.tp, t.direction, t.price);
                activeMarkers.push({{
                    time: t.time,
                    position: t.direction === 1 ? 'belowBar' : 'aboveBar',
                    color: t.direction === 1 ? '#2962ff' : '#ff9800',
                    shape: t.direction === 1 ? 'arrowUp' : 'arrowDown',
                    text: (t.direction === 1 ? 'Buy' : 'Sell')
                }});
                markersChanged = true;
            }} else if (t.type === 'exit') {{
                activeTrade = null;
                clearPriceLines();
                hideTradePanel();
                activeMarkers.push({{
                    time: t.time,
                    position: t.pnl > 0 ? 'aboveBar' : 'belowBar',
                    color: t.pnl > 0 ? '#4CAF50' : '#FF5252',
                    shape: 'circle',
                    text: (t.reason === 'tp' ? 'TP' : t.reason === 'sl' ? 'SL' : 'REV') + ' ' + (t.pnl*100).toFixed(1) + '%'
                }});
                markersChanged = true;
            }}
        }}
        if (markersChanged) {{
            candleSeries.setMarkers(activeMarkers.slice().sort((a,b) => a.time - b.time));
        }}
        currentIndex++;
        updateProgress();
    }}

    function pausePlayback() {{
        clearInterval(timer);
        isPlaying = false;
    }}

    document.getElementById('playBtn').addEventListener('click', () => {{
        if (!isPlaying) {{
            isPlaying = true;
            const speed = Math.max(1, parseInt(document.getElementById('speed').value) || 30);
            timer = setInterval(updateChart, speed);
        }}
    }});
    document.getElementById('pauseBtn').addEventListener('click', pausePlayback);
    document.getElementById('stepBtn').addEventListener('click', () => {{ pausePlayback(); updateChart(); }});
    document.getElementById('resetBtn').addEventListener('click', () => {{
        pausePlayback();
        currentIndex = 0;
        activeMarkers = [];
        activeTrade = null;
        clearPriceLines();
        hideTradePanel();
        candleSeries.setData([]);
        shortMaSeries.setData([]);
        longMaSeries.setData([]);
        smaFilterSeries.setData([]);
        // Re-run warmup silently
        for (let i = 0; i < Math.min(PREFILL, total); i++) updateChart();
        restoreActivePriceLines();     // Redraw lines for any open trade at warmup end
        chart.timeScale().fitContent();
    }});

    // Pre-fill warm-up bars (= long_window) silently
    for (let i = 0; i < Math.min(PREFILL, total); i++) updateChart();
    restoreActivePriceLines();         // Ensure SL/TP visible at start if a trade is already open
    chart.timeScale().fitContent();
</script>
</body>
</html>"""

    with open('replay.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("Success! Open replay.html to view the interactive replay engine.")


if __name__ == "__main__":
    main()
