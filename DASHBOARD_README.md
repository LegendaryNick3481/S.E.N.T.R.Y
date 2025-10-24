# Terminal Dashboard - Live Trading Display

A beautiful, live-updating terminal dashboard for the S.E.N.T.R.Y trading system.

## Features

✅ **Real-time price updates** - Live stock prices with color-coded changes
✅ **Sentiment analysis** - ML-based news sentiment scores
✅ **Mismatch detection** - Discord scores highlighting mismatched energy
✅ **Trading signals** - BUY/SELL/HOLD recommendations
✅ **Recent news** - Top news items with relevance scores
✅ **System metrics** - Websocket status, news count, errors
✅ **Live logs** - Recent system events and warnings
✅ **Static display** - No scrolling, everything updates in place

## Dashboard Layout

```
╭─────────────────────── SENTRY SYSTEM ──────────────────────╮
│ Status: LIVE | Market: OPEN | Uptime: 00:05:23             │
╰─────────────────────────────────────────────────────────────╯

┌─────────────── Symbol Analysis ──────────┬─── Metrics ───┐
│ Symbol   Price    Chg%   Sent  Discord   │ WebSocket: ✓  │
│ RELIANCE 2,450   +2.3%   0.85   0.45     │ News: 45      │
│ TCS      3,200   -1.1%  -0.30   0.62     │ Signals: 2    │
├────────────── Recent News ───────────────┼─── Logs ──────┤
│ [15:20] RELIANCE (0.92) Breaking news... │ INFO: Started │
│ [15:18] TCS (0.78) Analyst downgrade...  │ WARN: Mismatch│
└──────────────────────────────────────────┴───────────────┘
```

## Usage

### Run with live trading
```bash
python run_clean.py --mode live --watchlist RELIANCE TCS INFY
```

### Test the dashboard with simulated data
```bash
python test_dashboard.py
```

## Dashboard Components

### Header
- **Status**: System state (Initializing, LIVE, ERROR)
- **Market**: Market open/closed status
- **Uptime**: Time since system started
- **Current Time**: Live clock

### Symbol Analysis Table
Columns:
- **Symbol**: Stock ticker
- **Price**: Current price in ₹
- **Chg%**: Percentage change (green=up, red=down)
- **Sent**: Sentiment score from news (-1 to +1)
- **Rel**: Relevance score (how relevant news is)
- **Discord**: Mismatch score (higher = more mismatch)
- **Signal**: Trading signal (BUY/SELL/HOLD)

### Recent News
- Shows last 5 news items
- Format: `[Time] SYMBOL (Relevance) Title...`
- Color-coded by relevance

### System Metrics
- **WebSocket**: Connection status (✓/❌)
- **News Scraped**: Total news items fetched
- **Symbols Tracked**: Number of active symbols
- **Signals Generated**: Trading signals created
- **Errors**: Error count
- **Last Update**: Time of last data refresh

### System Logs
- Recent log messages (last 5)
- Color-coded by level:
  - **INFO** (cyan): Normal operations
  - **SUCCESS** (green): Successful actions
  - **WARNING** (yellow): Warnings and alerts
  - **ERROR** (red): Errors

## Color Coding

### Price Changes
- **Green**: Positive change
- **Red**: Negative change
- **Dim**: No change

### Sentiment
- **Green**: Positive (> 0.3)
- **Red**: Negative (< -0.3)
- **Yellow**: Neutral

### Discord Score
- **Yellow**: High mismatch (> 0.3)
- **White**: Normal

### Signals
- **Bold Green**: BUY
- **Bold Red**: SELL
- **Dim**: HOLD

## Integration

The dashboard automatically receives updates through the event bus:

```python
from utils.event_bus import event_bus

# Publish any event and it updates the dashboard
await event_bus.publish({
    "type": "price_update",
    "symbol": "RELIANCE",
    "price": 2450.50,
    "change": 2.3
})
```

## Event Types

The dashboard listens to these events:

- `status` - System status changes
- `price_update` - Live price updates
- `news` - News scraping completed
- `news_item` - Individual news item
- `sentiment` - Sentiment analysis result
- `mismatch` - Mismatch detection result
- `signals` - Trading signals generated
- `websocket_status` - WebSocket connection status
- `error` - Error occurred

## Keyboard Controls

- **Ctrl+C**: Exit the system gracefully

## Technical Details

- Built with `rich` library for terminal UI
- Runs in a separate thread for non-blocking display
- Updates at 1 FPS for smooth rendering
- All logs go to `sentry_dashboard.log` file
- Terminal is automatically restored on exit

## Troubleshooting

**Dashboard not updating:**
- Check that events are being published
- Verify event bus is imported correctly

**Garbled display:**
- Ensure terminal supports Unicode
- Try resizing the terminal window

**Exit doesn't restore terminal:**
- Use Ctrl+C (don't kill the process)
- If stuck, run: `reset` (Linux/Mac) or restart terminal (Windows)
