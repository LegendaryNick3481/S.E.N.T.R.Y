# Mismatched Energy Trading System

A sophisticated trading system that exploits the misalignment between news sentiment and price action in Indian markets. This system implements a "synesthesia detector" that identifies when market movements contradict news sentiment, creating profitable contrarian opportunities.

## 🎯 Core Concept

**Mismatched Energy**: When the market moves against what the news suggests, that's often the best play.

- **News is positive, price dropping** → Contrarian edge (potential buy)
- **News is negative, price rising** → Contrarian edge (potential sell)
- **Edge**: Exploiting misalignment between perception and reality

## 🏗️ System Architecture

### Core Components

1. **Data Layer** (`data/fyers_client.py`)
   - Fyers API integration for real-time market data
   - Historical OHLC data retrieval
   - Market microstructure analysis

2. **News Layer** (`news/news_scraper.py`)
   - RSS feed scraping (MoneyControl, ET, BSE/NSE)
   - Twitter/X news extraction
   - Real-time news sentiment analysis

3. **NLP Processing** (`nlp/sentiment_analyzer.py`)
   - Sentence transformers for embeddings
   - VADER sentiment analysis
   - Cross-modal correlation analysis

4. **Cross-Modal Analysis** (`analysis/cross_modal_analyzer.py`)
   - News-price mismatch detection
   - Discord scoring system
   - Volume and volatility anomaly detection

5. **Capital Allocation** (`scoring/capital_allocator.py`)
   - Position sizing based on discord scores
   - Risk management and portfolio optimization
   - Signal generation and execution

6. **Backtesting** (`backtesting/backtest_engine.py`)
   - Historical strategy validation
   - Performance metrics calculation
   - Risk-adjusted returns analysis

7. **Live Trading** (`trading/live_executor.py`)
   - Real-time signal generation
   - Fyers paper account integration
   - Automated trade execution

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Sentry

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env_example.txt .env

# Edit .env with your Fyers API credentials
FYERS_APP_ID=your_app_id_here
FYERS_SECRET_KEY=your_secret_key_here
FYERS_ACCESS_TOKEN=your_access_token_here
```

### 3. Usage Examples

#### Backtesting
```bash
# Run backtest for specific symbols
python main.py --mode backtest --symbols RELIANCE TCS INFY --start-date 2023-01-01 --end-date 2024-01-01
```

#### Live Trading
```bash
# Start live trading with watchlist
python main.py --mode live --watchlist RELIANCE TCS INFY HDFC ICICIBANK
```

#### Single Symbol Analysis
```bash
# Analyze specific symbols for mismatched energy
python main.py --mode analyze --symbols RELIANCE TCS
```

## 📊 How It Works

### 1. Data Collection
- **Market Data**: Real-time price, volume, order book from Fyers API
- **News Data**: RSS feeds from MoneyControl, ET, BSE/NSE announcements
- **Social Media**: Twitter/X scraping for market sentiment

### 2. Cross-Modal Analysis
```python
# Example workflow
news_sentiment = analyze_news_sentiment(symbol_news)
price_direction = calculate_price_movement(symbol_data)
discord_score = detect_mismatch(news_sentiment, price_direction)
```

### 3. Signal Generation
- **Discord Score**: Measures news-price misalignment
- **Confidence**: Based on sentiment strength and volume
- **Position Sizing**: Proportional to discord score and risk

### 4. Execution
- **Entry**: When discord score > threshold
- **Exit**: When mismatch resolves or stop-loss hit
- **Risk Management**: Position limits and portfolio exposure

## 🔧 Configuration

### Key Parameters

```python
# config.py
MIN_PRICE_MOVE_PERCENT = 2.0  # Minimum 2% move to consider
DISCORD_WEIGHTS = {
    'sentiment_price_correlation': 0.4,
    'volume_anomaly': 0.3,
    'volatility_spike': 0.3
}
MAX_POSITION_SIZE = 0.1  # 10% of capital per position
MAX_TOTAL_EXPOSURE = 0.5  # 50% total exposure
```

### News Sources
- MoneyControl RSS feeds
- Economic Times market news
- BSE/NSE announcements
- Twitter handles: @NSEIndia, @BSEIndia, @MoneyControlCom

## 📈 Performance Metrics

### Backtesting Results
- **Total Return**: Portfolio performance over time
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### Live Trading Metrics
- **Discord Score**: Real-time mismatch detection
- **Position Tracking**: Current holdings and P&L
- **Risk Metrics**: Portfolio exposure and concentration

## 🛡️ Risk Management

### Position Sizing
- Maximum 10% per position
- Total exposure limited to 50%
- Volatility-adjusted sizing

### Stop Losses
- Automatic position closure on mismatch resolution
- Portfolio-level risk limits
- Market hours enforcement

## 📁 Project Structure

```
Sentry/
├── main.py                 # Main orchestration system
├── config.py              # Configuration settings
├── requirements.txt        # Dependencies
├── data/
│   └── fyers_client.py    # Fyers API integration
├── news/
│   └── news_scraper.py    # News scraping system
├── nlp/
│   └── sentiment_analyzer.py # NLP processing
├── analysis/
│   └── cross_modal_analyzer.py # Cross-modal analysis
├── scoring/
│   └── capital_allocator.py # Capital allocation
├── backtesting/
│   └── backtest_engine.py # Backtesting framework
└── trading/
    └── live_executor.py   # Live trading execution
```

## 🔍 Example Workflow

### 1. Market Scan
```python
# System detects 2%+ price moves
significant_moves = detect_significant_moves(symbols)
# Result: [('RELIANCE', 3.2%), ('TCS', -2.8%)]
```

### 2. News Analysis
```python
# Scrape recent news for moving symbols
news_data = scrape_news(moving_symbols)
# Analyze sentiment
sentiment = analyze_sentiment(news_data)
# Result: RELIANCE: positive sentiment, TCS: negative sentiment
```

### 3. Cross-Modal Detection
```python
# Detect mismatches
for symbol, price_change in significant_moves:
    news_sentiment = get_sentiment(symbol)
    if news_sentiment > 0.1 and price_change < -0.02:
        # Positive news but price down → BUY signal
        generate_signal(symbol, 'BUY', discord_score)
```

### 4. Execution
```python
# Generate and execute trades
signals = generate_trading_signals(discord_analysis)
execute_trades(signals)
```

## 🚨 Important Notes

### Paper Trading
- **Always start with paper trading** to validate the system
- Use Fyers paper account for testing
- Monitor performance before live trading

### Market Hours
- System only trades during market hours (9:15 AM - 3:30 PM IST)
- Automatic position closure at market close

### Risk Disclaimer
- This is a high-risk trading strategy
- Past performance doesn't guarantee future results
- Use proper risk management and position sizing

## 🔧 Troubleshooting

### Common Issues
1. **Fyers API Connection**: Check credentials and token validity
2. **News Scraping**: Verify internet connection and source availability
3. **NLP Models**: Ensure sufficient memory for sentence transformers
4. **Market Data**: Check symbol validity and market hours

### Logs
- System logs are written to `mismatched_energy.log`
- Check logs for detailed error information
- Use `--log-level DEBUG` for verbose output

## 📚 Dependencies

### Core Libraries
- `pandas`, `numpy`, `scipy` - Data processing
- `sentence-transformers` - NLP embeddings
- `vaderSentiment` - Sentiment analysis
- `fyers-apiv3` - Fyers API integration
- `asyncio`, `aiohttp` - Async processing

### News Sources
- `feedparser` - RSS feed parsing
- `snscrape` - Twitter scraping
- `beautifulsoup4` - HTML parsing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Use at your own risk.

---

**Remember**: The key to this strategy is identifying when the market's "color" (news sentiment) doesn't match its "sound" (price action). When they're out of sync, that's your edge! 🎵📈

