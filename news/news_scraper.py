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
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

logger = logging.getLogger(__name__)

class NewsScraper:
    def __init__(self):
        self.session = None
        self.news_cache = {}
        self.symbol_patterns = self._load_symbol_patterns()
        self.relevance_model = None
        self.symbol_embeddings = {}
        
    async def initialize(self):
        """Initialize async session and ML models"""
        self.session = aiohttp.ClientSession()
        
        # Initialize ML model for relevance scoring
        try:
            self.relevance_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Load dynamic symbol list and generate embeddings
            await self._load_dynamic_symbols()
            
            logger.info("ML relevance model initialized with dynamic symbols")
        except Exception as e:
            logger.warning(f"Could not initialize ML model: {e}. Falling back to pattern matching.")
    
    async def _load_dynamic_symbols(self):
        """Load symbols dynamically from NSE API and generate embeddings"""
        try:
            # Get top NSE stocks
            symbols = await self._get_nse_top_stocks()
            
            # Generate embeddings for each symbol
            for symbol in symbols:
                # Create rich context for each symbol
                symbol_context = await self._create_symbol_context(symbol)
                self.symbol_embeddings[symbol] = self.relevance_model.encode(symbol_context)
                
            logger.info(f"Loaded {len(symbols)} symbols with ML embeddings")
            
        except Exception as e:
            logger.warning(f"Could not load dynamic symbols: {e}. Using fallback patterns.")
            # Fallback to static patterns
            for symbol, patterns in self.symbol_patterns.items():
                symbol_text = f"{symbol} {' '.join(patterns)}"
                self.symbol_embeddings[symbol] = self.relevance_model.encode(symbol_text)
    
    async def _get_nse_top_stocks(self, limit: int = None) -> List[str]:
        """Load stocks from tickers file"""
        try:
            from data.tickers import get_tickers
            
            symbols = get_tickers()
            
            if limit:
                symbols = symbols[:limit]
                
            logger.info(f"Loaded {len(symbols)} symbols: {symbols}")
            return symbols
            
        except Exception as e:
            logger.error(f"Error loading stocks from tickers: {e}")
            # Fallback to minimal list
            return ['HINDZINC', 'MANKIND', 'INDUSTOWER', 'DEEPINDS', 'FMGOETZE']
    
    async def _create_symbol_context(self, symbol: str) -> str:
        """Create rich context for symbol to improve ML matching"""
        # Simple context based on symbol name
        context = f"{symbol} company stock"
        return context
        
    async def close(self):
        """Close async session"""
        if self.session:
            await self.session.close()
    
    def _load_symbol_patterns(self) -> Dict[str, List[str]]:
        """Load symbol patterns for fallback filtering (ML is primary)"""
        # Minimal fallback patterns - ML handles the heavy lifting
        return {
            'RELIANCE': ['reliance', 'ril'],
            'TCS': ['tcs', 'tata consultancy'],
            'INFY': ['infosys', 'infy'],
            'HDFC': ['hdfc', 'hdfc bank'],
            'ICICIBANK': ['icici', 'icici bank'],
            'SBIN': ['sbi', 'state bank'],
            'BHARTIARTL': ['bharti', 'airtel'],
            'ITC': ['itc'],
            'KOTAKBANK': ['kotak', 'kotak bank'],
            'LT': ['larsen', 'l&t'],
            'ASIANPAINT': ['asian paint'],
            'MARUTI': ['maruti', 'maruti suzuki'],
            'NESTLEIND': ['nestle'],
            'TITAN': ['titan'],
            'ULTRACEMCO': ['ultracemco', 'ultra tech']
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
        """Get recent news for specific symbols from all sources"""
        symbol_news = {symbol: [] for symbol in symbols}
        
        try:
            # Get all news sources
            rss_news = await self.scrape_rss_feeds()
            twitter_news = await self.scrape_twitter_news(hours_back)
            reddit_news = await self.scrape_reddit_news(hours_back)
            announcements = await self.get_market_announcements(hours_back)
            
            # Get Google News for each symbol
            google_news = []
            for symbol in symbols:
                symbol_google_news = await self.scrape_google_news(symbol, hours_back)
                google_news.extend(symbol_google_news)
            
            all_news = rss_news + twitter_news + reddit_news + announcements + google_news
            
            # Filter news by symbols with ML-first relevance scoring
            for news_item in all_news:
                for symbol in symbols:
                    if self._is_news_relevant(news_item, symbol):
                        # Always add relevance score for ML-based filtering
                        if self.relevance_model and symbol in self.symbol_embeddings:
                            news_item['relevance_score'] = self._calculate_relevance_score(news_item, symbol)
                        else:
                            news_item['relevance_score'] = 1.0  # Fallback confidence
                        
                        # Add symbol context for better analysis
                        news_item['matched_symbol'] = symbol
                        symbol_news[symbol].append(news_item)
                        
        except Exception as e:
            logger.error(f"Error getting recent news: {e}")
            
        return symbol_news
    
    def _calculate_relevance_score(self, news_item: Dict, symbol: str) -> float:
        """Calculate ML-based relevance score between news and symbol"""
        if not self.relevance_model or symbol not in self.symbol_embeddings:
            return 0.0
            
        news_text = f"{news_item['title']} {news_item['description']}"
        news_embedding = self.relevance_model.encode(news_text)
        symbol_embedding = self.symbol_embeddings[symbol]
        
        similarity = cosine_similarity([news_embedding], [symbol_embedding])[0][0]
        return float(similarity)
    
    def _is_news_relevant(self, news_item: Dict, symbol: str, threshold: float = 0.3) -> bool:
        """ML-first relevance check with intelligent fallback"""
        # Primary: ML-based relevance scoring
        if self.relevance_model and symbol in self.symbol_embeddings:
            ml_score = self._calculate_relevance_score(news_item, symbol)
            if ml_score > threshold:
                logger.debug(f"ML match: {symbol} score={ml_score:.3f}")
                return True
        
        # Fallback: Enhanced pattern matching
        text = news_item['text'].lower()
        
        # Direct symbol mention (high confidence)
        if symbol.lower() in text:
            logger.debug(f"Direct match: {symbol}")
            return True
            
        # NSE/BSE symbol format (high confidence)
        if f"{symbol}.ns" in text or f"{symbol}.bo" in text:
            logger.debug(f"Exchange format match: {symbol}")
            return True
        
        # Pattern matching (medium confidence)
        if symbol in self.symbol_patterns:
            for pattern in self.symbol_patterns[symbol]:
                if pattern.lower() in text:
                    logger.debug(f"Pattern match: {symbol} -> {pattern}")
                    return True
        
        # No match found
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
            async with self.session.get(bse_url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Parse BSE announcements table
                    table = soup.find('table', {'id': 'ContentPlaceHolder1_grdAnn'})
                    if table:
                        for row in table.find_all('tr')[1:]:  # Skip header
                            cells = row.find_all('td')
                            if len(cells) >= 4:
                                announcement = {
                                    'company': cells[0].get_text().strip(),
                                    'subject': cells[1].get_text().strip(),
                                    'date': cells[2].get_text().strip(),
                                    'link': cells[3].find('a')['href'] if cells[3].find('a') else '',
                                    'source': 'BSE',
                                    'type': 'announcement',
                                    'text': self._clean_text(f"{cells[0].get_text()} {cells[1].get_text()}")
                                }
                                announcements.append(announcement)
            
            # NSE announcements
            nse_url = "https://www.nseindia.com/corporates-actions"
            async with self.session.get(nse_url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Parse NSE announcements
                    table = soup.find('table', class_='dataTable')
                    if table:
                        for row in table.find_all('tr')[1:]:  # Skip header
                            cells = row.find_all('td')
                            if len(cells) >= 3:
                                announcement = {
                                    'company': cells[0].get_text().strip(),
                                    'subject': cells[1].get_text().strip(),
                                    'date': cells[2].get_text().strip(),
                                    'link': cells[1].find('a')['href'] if cells[1].find('a') else '',
                                    'source': 'NSE',
                                    'type': 'announcement',
                                    'text': self._clean_text(f"{cells[0].get_text()} {cells[1].get_text()}")
                                }
                                announcements.append(announcement)
                    
        except Exception as e:
            logger.error(f"Error getting market announcements: {e}")
            
        return announcements
    
    async def scrape_reddit_news(self, hours_back: int = 24) -> List[Dict]:
        """Scrape Reddit for market sentiment from Indian finance subreddits"""
        news_items = []
        
        try:
            for subreddit in Config.NEWS_SOURCES['reddit']:
                # Use Reddit JSON API with higher limit for better coverage
                url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=50"
                async with self.session.get(url, headers={'User-Agent': 'SENTRY/1.0'}) as response:
                    if response.status == 200:
                        data = await response.json()
                        for post in data['data']['children']:
                            post_data = post['data']
                            
                            # Filter by time and quality
                            post_time = datetime.fromtimestamp(post_data['created_utc'])
                            if post_time >= datetime.now() - timedelta(hours=hours_back):
                                # Quality filters for Indian finance subreddits
                                score = post_data.get('score', 0)
                                comments = post_data.get('num_comments', 0)
                                
                                # Skip low-quality posts (adjust thresholds per subreddit)
                                if subreddit in ['IndianStockMarket', 'IndianStreetBets']:
                                    min_score = 2  # More lenient for active trading subs
                                else:
                                    min_score = 1  # Stricter for investment subs
                                
                                if score >= min_score or comments >= 3:
                                    # Extract stock mentions from title and content
                                    full_text = f"{post_data['title']} {post_data.get('selftext', '')}"
                                    stock_mentions = self._extract_stock_mentions(full_text)
                                    
                                    news_item = {
                                        'title': post_data['title'],
                                        'description': post_data.get('selftext', ''),
                                        'link': f"https://reddit.com{post_data['permalink']}",
                                        'published': post_time,
                                        'source': f'r/{subreddit}',
                                        'type': 'reddit',
                                        'text': self._clean_text(full_text),
                                        'score': score,
                                        'comments': comments,
                                        'upvote_ratio': post_data.get('upvote_ratio', 0),
                                        'stock_mentions': stock_mentions,
                                        'subreddit_type': self._get_subreddit_type(subreddit)
                                    }
                                    news_items.append(news_item)
                                
        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")
            
        return news_items
    
    def _extract_stock_mentions(self, text: str) -> List[str]:
        """Extract potential stock symbols from Reddit text"""
        mentions = []
        text_upper = text.upper()
        
        # Common Indian stock patterns
        patterns = [
            r'\b[A-Z]{3,6}\b',  # 3-6 letter symbols
            r'\b[A-Z]{2,4}\b',  # 2-4 letter symbols
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                # Filter out common words that aren't stocks
                if match not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'HIS', 'HAS', 'HAD', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE']:
                    if len(match) >= 2 and match not in mentions:
                        mentions.append(match)
        
        return mentions[:5]  # Limit to top 5 mentions
    
    def _get_subreddit_type(self, subreddit: str) -> str:
        """Categorize subreddit by focus area"""
        subreddit_types = {
            'IndianStockMarket': 'retail_trading',
            'IndiaInvestments': 'long_term_investing', 
            'NSEIndia': 'stock_specific',
            'IndianStreetBets': 'speculative_trading',
            'IndiaFinance': 'macro_economics'
        }
        return subreddit_types.get(subreddit, 'general')
    
    async def update_symbol_list(self, new_symbols: List[str]):
        """Dynamically update symbol list and regenerate embeddings"""
        try:
            if not self.relevance_model:
                logger.warning("ML model not available for dynamic symbol updates")
                return False
            
            # Add new symbols to embeddings
            for symbol in new_symbols:
                if symbol not in self.symbol_embeddings:
                    symbol_context = await self._create_symbol_context(symbol)
                    self.symbol_embeddings[symbol] = self.relevance_model.encode(symbol_context)
                    logger.info(f"Added new symbol: {symbol}")
            
            logger.info(f"Updated symbol list with {len(new_symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error updating symbol list: {e}")
            return False
    
    async def get_relevant_symbols_for_news(self, news_item: Dict, threshold: float = 0.3) -> List[str]:
        """Find all symbols relevant to a news item using ML scoring"""
        relevant_symbols = []
        
        if not self.relevance_model:
            return relevant_symbols
        
        try:
            for symbol in self.symbol_embeddings.keys():
                score = self._calculate_relevance_score(news_item, symbol)
                if score > threshold:
                    relevant_symbols.append((symbol, score))
            
            # Sort by relevance score
            relevant_symbols.sort(key=lambda x: x[1], reverse=True)
            return [symbol for symbol, score in relevant_symbols]
            
        except Exception as e:
            logger.error(f"Error finding relevant symbols: {e}")
            return relevant_symbols
    
    async def scrape_google_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Scrape Google News for specific symbol"""
        news_items = []
        
        try:
            # Construct Google News RSS URL
            query = Config.NEWS_SOURCES['google_news']['query_params'].format(symbol=symbol)
            url = f"{Config.NEWS_SOURCES['google_news']['base_url']}?{query}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries:
                        # Filter by time
                        entry_time = self._parse_date(entry.get('published', ''))
                        if entry_time >= datetime.now() - timedelta(hours=hours_back):
                            news_item = {
                                'title': entry.get('title', ''),
                                'description': entry.get('summary', ''),
                                'link': entry.get('link', ''),
                                'published': entry_time,
                                'source': 'Google News',
                                'type': 'google_news',
                                'text': self._clean_text(f"{entry.get('title', '')} {entry.get('summary', '')}")
                            }
                            news_items.append(news_item)
                            
        except Exception as e:
            logger.error(f"Error scraping Google News for {symbol}: {e}")
            
        return news_items

