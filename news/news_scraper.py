"""
News scraping system for Indian markets
Supports RSS feeds and Twitter/X scraping
"""
import asyncio
import aiohttp
import feedparser
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import re
from bs4 import BeautifulSoup
import logging
from config import Config

logger = logging.getLogger(__name__)

class NewsScraper:
    def __init__(self):
        self.session = None
        self.news_cache = {}
        self.symbol_patterns = self._load_symbol_patterns()
        
    async def initialize(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        """Close async session"""
        if self.session:
            await self.session.close()
    
    def _load_symbol_patterns(self) -> Dict[str, List[str]]:
        """Load symbol patterns for news filtering"""
        # Common Indian stock symbols and their variations
        return {
            'RELIANCE': ['reliance', 'ril', 'reliance industries'],
            'TCS': ['tcs', 'tata consultancy', 'tata consultancy services'],
            'INFY': ['infosys', 'infy'],
            'HDFC': ['hdfc', 'hdfc bank', 'hdfc ltd'],
            'ICICIBANK': ['icici', 'icici bank'],
            'SBIN': ['sbi', 'state bank', 'state bank of india'],
            'BHARTIARTL': ['bharti', 'airtel', 'bharti airtel'],
            'ITC': ['itc ltd', 'itc'],
            'KOTAKBANK': ['kotak', 'kotak bank', 'kotak mahindra'],
            'LT': ['larsen', 'l&t', 'larsen & toubro'],
            'ASIANPAINT': ['asian paint', 'asian paints'],
            'MARUTI': ['maruti', 'maruti suzuki'],
            'NESTLEIND': ['nestle', 'nestle india'],
            'TITAN': ['titan', 'titan company'],
            'ULTRACEMCO': ['ultracemco', 'ultra tech', 'ultratech cement']
        }
    
    async def scrape_rss_feeds(self) -> List[Dict]:
        """Scrape RSS feeds for market news"""
        news_items = []
        
        try:
            for rss_url in Config.NEWS_SOURCES['rss']:
                try:
                    async with self.session.get(rss_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            feed = feedparser.parse(content)
                            
                            for entry in feed.entries:
                                # Parse entry
                                news_item = {
                                    'title': entry.get('title', ''),
                                    'description': entry.get('description', ''),
                                    'link': entry.get('link', ''),
                                    'published': self._parse_date(entry.get('published', '')),
                                    'source': rss_url,
                                    'type': 'rss'
                                }
                                
                                # Clean and extract text
                                news_item['text'] = self._clean_text(
                                    f"{news_item['title']} {news_item['description']}"
                                )
                                
                                news_items.append(news_item)
                                
                except Exception as e:
                    logger.error(f"Error scraping RSS feed {rss_url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in RSS scraping: {e}")
            
        return news_items
    
    async def scrape_twitter_news(self, hours_back: int = 24) -> List[Dict]:
        """Scrape Twitter/X for market news"""
        news_items = []
        
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            for handle in Config.NEWS_SOURCES['twitter_handles']:
                try:
                    # Create search query
                    query = f"from:{handle} since:{start_time.strftime('%Y-%m-%d')} until:{end_time.strftime('%Y-%m-%d')}"
                    
                    # Scrape tweets
                    tweets = sntwitter.TwitterSearchScraper(query).get_items()
                    
                    for tweet in tweets:
                        if tweet.date >= start_time:
                            news_item = {
                                'title': tweet.content[:100] + '...' if len(tweet.content) > 100 else tweet.content,
                                'description': tweet.content,
                                'link': tweet.url,
                                'published': tweet.date,
                                'source': handle,
                                'type': 'twitter',
                                'text': self._clean_text(tweet.content)
                            }
                            news_items.append(news_item)
                            
                except Exception as e:
                    logger.error(f"Error scraping Twitter handle {handle}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in Twitter scraping: {e}")
            
        return news_items
    
    async def get_recent_news(self, symbols: List[str], hours_back: int = 1) -> Dict[str, List[Dict]]:
        """Get recent news for specific symbols"""
        symbol_news = {symbol: [] for symbol in symbols}
        
        try:
            # Get all news
            rss_news = await self.scrape_rss_feeds()
            twitter_news = await self.scrape_twitter_news(hours_back)
            all_news = rss_news + twitter_news
            
            # Filter news by symbols
            for news_item in all_news:
                for symbol in symbols:
                    if self._is_news_relevant(news_item, symbol):
                        symbol_news[symbol].append(news_item)
                        
        except Exception as e:
            logger.error(f"Error getting recent news: {e}")
            
        return symbol_news
    
    def _is_news_relevant(self, news_item: Dict, symbol: str) -> bool:
        """Check if news is relevant to a symbol"""
        text = news_item['text'].lower()
        
        # Check symbol patterns
        if symbol in self.symbol_patterns:
            for pattern in self.symbol_patterns[symbol]:
                if pattern.lower() in text:
                    return True
        
        # Check for direct symbol mention
        if symbol.lower() in text:
            return True
            
        # Check for NSE/BSE symbol format
        if f"{symbol}.ns" in text or f"{symbol}.bo" in text:
            return True
            
        return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text.strip()
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        try:
            # Try common date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%SZ'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # Fallback to current time
            return datetime.now()
            
        except Exception:
            return datetime.now()
    
    async def get_news_sentiment_summary(self, symbol: str, hours_back: int = 1) -> Dict:
        """Get sentiment summary for a symbol's news"""
        try:
            symbol_news = await self.get_recent_news([symbol], hours_back)
            news_items = symbol_news.get(symbol, [])
            
            if not news_items:
                return {
                    'sentiment_score': 0.0,
                    'news_count': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'recent_news': []
                }
            
            # This will be enhanced with actual sentiment analysis
            # For now, return basic structure
            return {
                'sentiment_score': 0.0,  # Will be calculated by NLP processor
                'news_count': len(news_items),
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': len(news_items),
                'recent_news': news_items[:5]  # Last 5 news items
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment summary: {e}")
            return {
                'sentiment_score': 0.0,
                'news_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'recent_news': []
            }
    
    async def get_market_announcements(self, hours_back: int = 24) -> List[Dict]:
        """Get market announcements from BSE/NSE"""
        announcements = []
        
        try:
            # BSE announcements
            bse_url = "https://www.bseindia.com/corporates/ann.aspx"
            async with self.session.get(bse_url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Parse announcements (this would need to be customized based on BSE's structure)
                    # For now, return empty list
                    pass
            
            # NSE announcements
            nse_url = "https://www.nseindia.com/corporates-actions"
            async with self.session.get(nse_url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Parse announcements (this would need to be customized based on NSE's structure)
                    # For now, return empty list
                    pass
                    
        except Exception as e:
            logger.error(f"Error getting market announcements: {e}")
            
        return announcements

