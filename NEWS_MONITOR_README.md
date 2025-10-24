# News Monitor - Terminal-based News Display

A clean, terminal-based news monitoring tool for the Mismatched Energy trading system.

## Usage

### Display news for specific symbols (one-time)
```bash
python news_monitor.py --symbols RELIANCE TCS INFY --hours 1
```

### Display news for all symbols from tickers (one-time)
```bash
python news_monitor.py --hours 1
```

### Continuous monitoring with auto-refresh
```bash
python news_monitor.py --continuous --refresh 5
```

### Display all news from all sources (no symbol filtering)
```bash
python news_monitor.py --mode all --hours 24
```

## Options

- `--mode`: Display mode
  - `symbols` (default): Filter news by symbols
  - `all`: Show all news from all sources
  
- `--symbols`: Specific symbols to monitor (space-separated)
  - If not provided, loads from `data/tickers.py`
  
- `--hours`: Hours to look back for news (default: 1)

- `--continuous`: Enable continuous monitoring with auto-refresh

- `--refresh`: Refresh interval in minutes for continuous mode (default: 5)

## Features

✅ **Clean terminal display** with color-coded output
✅ **Multiple sources**: RSS feeds, Reddit, Google News
✅ **Symbol filtering**: See only relevant news
✅ **Relevance scoring**: ML-based ranking
✅ **Continuous mode**: Auto-refresh at intervals
✅ **All sources mode**: View everything being scraped
✅ **Reddit metrics**: Shows upvotes, comments, ratios

## Examples

```bash
# Quick check for top 5 symbols
python news_monitor.py --symbols RELIANCE TCS INFY HDFCBANK SBIN --hours 2

# Monitor all tickers continuously (refresh every 10 min)
python news_monitor.py --continuous --refresh 10

# See all news from past 24 hours
python news_monitor.py --mode all --hours 24
```
