"""
Configuration settings for Mismatched Energy trading system
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Fyers API Configuration
    FYERS_APP_ID = os.getenv('FYERS_APP_ID')
    FYERS_SECRET_KEY = os.getenv('FYERS_SECRET_KEY')
    FYERS_REDIRECT_URI = os.getenv('FYERS_REDIRECT_URI', 'https://trade.fyers.in/api-login/redirect-uri')
    FYERS_ACCESS_TOKEN = os.getenv('FYERS_ACCESS_TOKEN')
    
    # News Sources (Indian Markets)
    NEWS_SOURCES = {
        'rss': [
            'https://www.moneycontrol.com/rss/business.xml',
            'https://www.moneycontrol.com/rss/marketnews.xml',
            'https://economictimes.indiatimes.com/markets/rssfeeds/1977029391.cms',
            'https://www.bseindia.com/rss/feeds/ann.xml',
            'https://www.nseindia.com/rss/feeds/ann.xml',
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.investing.com/rss/news.rss'
        ],
        'twitter_handles': [
            '@NSEIndia',
            '@BSEIndia', 
            '@MoneyControlCom',
            '@EconomicTimes',
            '@livemint'
        ],
        'reddit': [
            'IndianStockMarket',      # Retail traders, daily sentiment, live market memes
            'IndiaInvestments',       # Long-term investing, mutual funds, company deep-dives
            'NSEIndia',              # Stock-specific discussions, result reactions
            'IndianStreetBets',      # Meme-style trades, speculative buzz
            'IndiaFinance'           # RBI, policy, macroeconomic sentiment
        ],
        'google_news': {
            'base_url': 'https://news.google.com/rss/search',
            'query_params': 'q={symbol}+stock+news&hl=en&gl=IN&ceid=IN:en'
        }
    }
    
    # Market Configuration
    MARKET_HOURS = {
        'start': '09:15',
        'end': '15:30',
        'timezone': 'Asia/Kolkata'
    }
    
    # Trading Parameters
    MIN_PRICE_MOVE_PERCENT = 2.0  # Minimum 2% move to consider
    LOOKBACK_PERIODS = {
        'price': 5,  # 5 seconds for price data
        'news': 300,  # 5 minutes for news
        'correlation': 60  # 1 minute for correlation analysis
    }
    
    # NLP Configuration
    SENTENCE_TRANSFORMER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    SENTIMENT_THRESHOLD = 0.1  # Minimum sentiment score to consider
    
    # Discord Scoring
    DISCORD_WEIGHTS = {
        'sentiment_price_correlation': 0.4,
        'volume_anomaly': 0.3,
        'volatility_spike': 0.3
    }
    
    # Capital Allocation
    MAX_POSITION_SIZE = 0.1  # 10% of capital per position
    MAX_TOTAL_EXPOSURE = 0.5  # 50% total exposure
    MIN_CORRELATION_THRESHOLD = -0.3  # Minimum negative correlation for discord
    
    # Backtesting
    BACKTEST_START_DATE = '2023-01-01'
    BACKTEST_END_DATE = '2024-01-01'
    INITIAL_CAPITAL = 100000  # ₹1,00,000
    
    # Logging
    LOG_LEVEL = 'WARNING'  # Suppress INFO and DEBUG messages
    LOG_FILE = 'mismatched_energy.log'

