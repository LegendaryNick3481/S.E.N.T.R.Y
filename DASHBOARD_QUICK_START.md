# Terminal Dashboard - Quick Start

## What Changed?

✅ **`run_clean.py`** - Now shows a live, static terminal dashboard instead of scrolling logs
✅ **Event bus integration** - Dashboard updates automatically from trading system events
✅ **Real-time display** - Prices, sentiment, news, signals all update in place

## How to Use

### 1. Test the Dashboard (Simulated Data)
```bash
python test_dashboard.py
```
This will show you the dashboard with fake data updating in real-time.

### 2. Run with Live Trading
```bash
python run_clean.py --mode live --watchlist RELIANCE TCS INFY
```
The dashboard will show:
- Live prices from websocket
- News as it's scraped
- Sentiment analysis results
- Mismatch detection (discord scores)
- Trading signals (BUY/SELL/HOLD)

### 3. Independent News Monitor
```bash
# See all news from all sources
python news_monitor.py --mode all --hours 24

# Monitor specific symbols continuously
python news_monitor.py --symbols RELIANCE TCS INFY --continuous --refresh 5
```

## Dashboard Features

### What You'll See:

**Header**
- System status, market status, uptime, current time

**Symbol Analysis Table**
- Symbol, Price, Change%, Sentiment, Relevance, Discord, Signal
- Color-coded (green=positive, red=negative)

**Recent News**
- Last 5 news items with relevance scores

**System Metrics**
- WebSocket status
- News scraped count
- Signals generated
- Error count

**System Logs**
- Recent events (INFO, WARNING, ERROR, SUCCESS)

## Key Improvements

### Before:
```
2025-01-24 15:23:45 - INFO - Starting system...
2025-01-24 15:23:46 - INFO - Connected to websocket
2025-01-24 15:23:47 - INFO - Scraped 45 news items
2025-01-24 15:23:48 - INFO - Detected mismatch: RELIANCE
... (logs keep scrolling)
```

### After:
```
╭──────────── SENTRY SYSTEM ────────────╮
│ Status: LIVE | Market: OPEN           │
╰────────────────────────────────────────╯
┌──── Symbol Analysis ─────┬─ Metrics ─┐
│ RELIANCE  2,450  +2.3%   │ News: 45  │
│ Discord: 0.45 → BUY      │ WS: ✓     │
└──────────────────────────┴───────────┘
(Everything updates in place, no scrolling)
```

## Files Created/Modified

### New Files:
- `news_monitor.py` - Standalone news monitoring tool
- `test_dashboard.py` - Dashboard test with simulated data
- `DASHBOARD_README.md` - Detailed documentation
- `NEWS_MONITOR_README.md` - News monitor documentation

### Modified Files:
- `run_clean.py` - Now includes live dashboard
- `utils/event_bus.py` - Added dashboard callback support
- `data/fyers_client.py` - Publishes price update events
- `trading/live_executor.py` - Publishes news item events

## Dependencies Added

```bash
pip install rich colorama
```

## Tips

1. **Terminal Size**: Dashboard works best with a terminal at least 120 columns wide
2. **Logging**: All logs go to `sentry_dashboard.log` instead of terminal
3. **Exit**: Always use Ctrl+C to exit gracefully
4. **Colors**: If colors don't work, ensure your terminal supports ANSI codes

## Troubleshooting

**Dashboard looks garbled?**
- Make terminal window larger (120+ columns recommended)
- Use Windows Terminal instead of CMD (better Unicode support)

**Not updating?**
- Check that event_bus is publishing events
- Look at `sentry_dashboard.log` for errors

**Want old scrolling logs back?**
- Use `python main.py` instead of `python run_clean.py`

