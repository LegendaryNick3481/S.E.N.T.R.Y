# Usage Guide - Mismatched Energy Trading System

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py

# Test the system
python test_system.py
```

### 2. Configuration
Edit `.env` file with your Fyers API credentials:
```bash
FYERS_APP_ID=your_app_id_here
FYERS_SECRET_KEY=your_secret_key_here
FYERS_ACCESS_TOKEN=your_access_token_here
```

## 📊 Usage Examples

### Backtesting
```bash
# Basic backtest
python main.py --mode backtest --symbols RELIANCE TCS INFY

# Custom date range
python main.py --mode backtest --symbols RELIANCE TCS --start-date 2023-01-01 --end-date 2023-12-31

# Large watchlist
python main.py --mode backtest --symbols RELIANCE TCS INFY HDFC ICICIBANK SBIN BHARTIARTL
```

### Live Trading
```bash
# Start live trading
python main.py --mode live --watchlist RELIANCE TCS INFY

# Paper trading (recommended first)
python main.py --mode live --watchlist RELIANCE TCS
```

### Analysis Mode
```bash
# Analyze specific symbols
python main.py --mode analyze --symbols RELIANCE TCS

# Get market overview
python main.py --mode analyze --symbols RELIANCE TCS INFY HDFC ICICIBANK
```

## 🔧 Configuration Options

### Key Parameters (config.py)
```python
# Trading Parameters
MIN_PRICE_MOVE_PERCENT = 2.0  # Minimum 2% move to consider
MAX_POSITION_SIZE = 0.1       # 10% of capital per position
MAX_TOTAL_EXPOSURE = 0.5      # 50% total exposure

# Discord Scoring Weights
DISCORD_WEIGHTS = {
    'sentiment_price_correlation': 0.4,
    'volume_anomaly': 0.3,
    'volatility_spike': 0.3
}

# Market Hours
MARKET_HOURS = {
    'start': '09:15',
    'end': '15:30',
    'timezone': 'Asia/Kolkata'
}
```

## 📈 Understanding the Output

### Backtest Results
```
Mismatched Energy Strategy - Backtest Report
==========================================

Performance Summary:
- Initial Capital: ₹100,000.00
- Final Value: ₹125,000.00
- Total Return: 25.00%
- Annualized Return: 30.00%

Risk Metrics:
- Volatility: 15.20%
- Sharpe Ratio: 1.85
- Maximum Drawdown: 8.50%

Trading Activity:
- Total Trades: 45
- Win Rate: 65.00%
```

### Live Trading Output
```
Processing trading cycle...
Detected 3 significant moves
Found 2 mismatched energy opportunities
Executing BUY signal for RELIANCE: 10 shares
Portfolio value: ₹125,000.00
```

### Analysis Output
```
Analysis for RELIANCE:
{
  "symbol": "RELIANCE",
  "sentiment_summary": {
    "weighted_sentiment": 0.25,
    "news_count": 5,
    "confidence": 0.8
  },
  "mismatch_analysis": {
    "discord_score": 0.65,
    "is_mismatched": true,
    "confidence": 0.75
  }
}
```

## 🎯 Strategy Logic

### When to Buy
- **News sentiment**: Positive (>0.1)
- **Price action**: Downward move (>2%)
- **Discord score**: High (>0.3)
- **Volume**: Above average
- **Confidence**: High (>0.7)

### When to Sell
- **News sentiment**: Negative (<-0.1)
- **Price action**: Upward move (>2%)
- **Discord score**: High (>0.3)
- **Volume**: Above average
- **Confidence**: High (>0.7)

### Risk Management
- **Position sizing**: Based on discord score and volatility
- **Stop loss**: Automatic on mismatch resolution
- **Portfolio limits**: Max 10% per position, 50% total exposure
- **Market hours**: Only trade during 9:15 AM - 3:30 PM IST

## 🔍 Monitoring and Debugging

### Logs
- System logs: `mismatched_energy.log`
- Backtest results: `backtest_results.json`
- Performance plots: `backtest_results.png`

### Key Metrics to Monitor
1. **Discord Score**: Higher = better opportunity
2. **Confidence**: Higher = more reliable signal
3. **Volume Anomaly**: Confirms price movement
4. **Volatility Spike**: Indicates market stress
5. **Portfolio Exposure**: Keep under 50%

### Common Issues
1. **No signals**: Check market hours and news sources
2. **Low confidence**: Verify news sentiment analysis
3. **API errors**: Check Fyers credentials and connection
4. **High drawdown**: Reduce position sizes or increase thresholds

## 📊 Performance Optimization

### Parameter Tuning
```python
# Increase sensitivity
MIN_PRICE_MOVE_PERCENT = 1.5  # Lower threshold
MIN_CORRELATION_THRESHOLD = -0.2  # More sensitive

# Decrease sensitivity
MIN_PRICE_MOVE_PERCENT = 3.0  # Higher threshold
MIN_CORRELATION_THRESHOLD = -0.5  # Less sensitive
```

### News Source Optimization
```python
# Add more sources
NEWS_SOURCES = {
    'rss': [
        'https://www.moneycontrol.com/rss/business.xml',
        'https://economictimes.indiatimes.com/markets/rssfeeds/1977029391.cms',
        # Add more sources
    ]
}
```

## 🚨 Safety Guidelines

### Paper Trading First
1. Always start with paper trading
2. Monitor performance for at least 1 week
3. Verify all signals before going live
4. Start with small position sizes

### Risk Management
1. Never risk more than you can afford to lose
2. Use proper position sizing
3. Monitor portfolio exposure
4. Set stop losses

### Market Conditions
1. Avoid trading during high volatility
2. Monitor market hours
3. Check for major news events
4. Adjust parameters based on market conditions

## 📞 Support

### Troubleshooting
1. Check logs for error messages
2. Verify API credentials
3. Test with demo script
4. Run system tests

### Getting Help
1. Check README.md for detailed documentation
2. Review config.py for parameter explanations
3. Run examples/demo.py for usage examples
4. Use test_system.py to verify setup

---

**Remember**: This is a sophisticated trading system that requires careful monitoring and risk management. Always start with paper trading and gradually increase exposure as you gain confidence in the system's performance.
