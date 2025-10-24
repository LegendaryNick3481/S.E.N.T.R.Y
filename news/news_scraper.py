"""
Enhanced news scraping system for Indian markets
Supports RSS feeds, web scraping with validated sources
Includes robust error handling and ML-based relevance scoring
"""
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import re
from bs4 import BeautifulSoup
import logging
from config import Config
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class NewsScraper:
    # Valid RSS feeds for Indian markets (verified 2025)
    VALID_RSS_FEEDS = {
        'moneycontrol': 'https://www.moneycontrol.com/rss/latestnews.xml',
        'moneycontrol_markets': 'https://www.moneycontrol.com/rss/marketreports.xml',
        'moneycontrol_ipo': 'https://www.moneycontrol.com/rss/ipo.xml',
        'economic_times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
        'business_standard': 'https://www.business-standard.com/rss/markets-106.rss',
        'livemint': 'https://www.livemint.com/rss/markets',
        'bse_notices': 'https://www.bseindia.com/data/xml/notices.xml',
        'bse_sensex': 'https://www.bseindia.com/data/xml/sensexrss.xml',
        'tradebrains': 'https://tradebrains.in/blog/feed/',
        'value_research': 'https://www.valueresearchonline.com/rss/',
    }

    # Reddit Indian finance communities
    REDDIT_SUBREDDITS = [
        'IndianStockMarket',
        'IndiaInvestments',
        'IndianStreetBets',
        'StockMarketIndia'
    ]

    def __init__(self):
        self.session = None
        self.news_cache = {}
        self.symbol_patterns = self._load_symbol_patterns()
        self.relevance_model = None
        self.symbol_embeddings = {}
        self.rate_limit_delay = 1.0  # Seconds between requests
        self.last_request_time = {}

    async def initialize(self):
        """Initialize async session and ML models"""
        # Don't reinitialize if already initialized
        if self.session is not None and not self.session.closed:
            logger.debug("Session already initialized, skipping...")
            return
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30, force_close=True)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

        # Initialize ML model for relevance scoring
        try:
            self.relevance_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            await self._load_dynamic_symbols()
            logger.info("ML relevance model initialized with dynamic symbols")
        except Exception as e:
            logger.warning(f"Could not initialize ML model: {e}. Falling back to pattern matching.")

    async def _load_dynamic_symbols(self):
        """Load symbols dynamically and generate embeddings"""
        try:
            symbols = await self._get_nse_top_stocks()

            for symbol in symbols:
                symbol_context = await self._create_symbol_context(symbol)
                self.symbol_embeddings[symbol] = self.relevance_model.encode(symbol_context)

            logger.info(f"Loaded {len(symbols)} symbols with ML embeddings")

        except Exception as e:
            logger.warning(f"Could not load dynamic symbols: {e}. Using fallback patterns.")
            for symbol, patterns in self.symbol_patterns.items():
                symbol_text = f"{symbol} {' '.join(patterns)}"
                if self.relevance_model:
                    self.symbol_embeddings[symbol] = self.relevance_model.encode(symbol_text)

    async def _get_nse_top_stocks(self, limit: int = None) -> List[str]:
        """Load stocks from tickers file"""
        try:
            from data.tickers import get_tickers
            symbols = get_tickers()

            if limit:
                symbols = symbols[:limit]

            logger.info(f"Loaded {len(symbols)} symbols")
            return symbols

        except Exception as e:
            logger.error(f"Error loading stocks from tickers: {e}")
            # Fallback to top stocks
            return ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
                    'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK']

    async def _create_symbol_context(self, symbol: str) -> str:
        """Create rich context for symbol to improve ML matching"""
        context = f"{symbol} stock share equity company India NSE BSE"

        # Add common name variations if available
        if symbol in self.symbol_patterns:
            context += " " + " ".join(self.symbol_patterns[symbol])

        return context

    async def _rate_limit_wait(self, source: str):
        """Implement rate limiting per source"""
        current_time = asyncio.get_event_loop().time()
        last_time = self.last_request_time.get(source, 0)

        time_since_last = current_time - last_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time[source] = asyncio.get_event_loop().time()

    async def close(self):
        """Close async session"""
        if self.session and not self.session.closed:
            await self.session.close()
            # Wait a bit for underlying connections to close
            await asyncio.sleep(0.250)
            self.session = None

    def _load_symbol_patterns(self) -> Dict[str, List[str]]:
        """Load symbol patterns for fallback filtering"""
        return {
            'RELIANCE': ['reliance', 'ril', 'reliance industries'],
            'TCS': ['tcs', 'tata consultancy', 'tata consultancy services'],
            'INFY': ['infosys', 'infy'],
            'HDFCBANK': ['hdfc', 'hdfc bank'],
            'ICICIBANK': ['icici', 'icici bank'],
            'SBIN': ['sbi', 'state bank', 'state bank of india'],
            'BHARTIARTL': ['bharti', 'airtel', 'bharti airtel'],
            'ITC': ['itc', 'itc limited'],
            'KOTAKBANK': ['kotak', 'kotak bank', 'kotak mahindra'],
            'LT': ['larsen', 'l&t', 'larsen toubro', 'larsen & toubro'],
            'HINDUNILVR': ['hindustan unilever', 'hul', 'hindustan lever'],
            'ASIANPAINT': ['asian paints', 'asian paint'],
            'MARUTI': ['maruti', 'maruti suzuki'],
            'NESTLEIND': ['nestle', 'nestle india'],
            'TITAN': ['titan', 'titan company'],
            'ULTRACEMCO': ['ultratech', 'ultra tech', 'ultratech cement'],
            'WIPRO': ['wipro', 'wipro limited'],
            'AXISBANK': ['axis', 'axis bank'],
            'BAJFINANCE': ['bajaj finance', 'bajaj fin'],
            'TATASTEEL': ['tata steel'],
        }

    async def scrape_rss_feeds(self, feed_names: List[str] = None) -> List[Dict]:
        """Scrape RSS feeds with retry logic and error handling"""
        # Ensure session is initialized and not closed
        if self.session is None or self.session.closed:
            logger.warning("Session not initialized or closed, initializing now...")
            await self.initialize()
        
        news_items = []

        feeds_to_scrape = feed_names or list(self.VALID_RSS_FEEDS.keys())

        for feed_name in feeds_to_scrape:
            if feed_name not in self.VALID_RSS_FEEDS:
                logger.warning(f"Unknown feed: {feed_name}")
                continue

            rss_url = self.VALID_RSS_FEEDS[feed_name]

            try:
                await self._rate_limit_wait(feed_name)

                retries = 3
                for attempt in range(retries):
                    try:
                        # Check session again before use
                        if self.session is None or self.session.closed:
                            logger.warning("Session closed during scraping, reinitializing...")
                            await self.initialize()
                        
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        }

                        async with self.session.get(rss_url, headers=headers) as response:
                            if response.status == 200:
                                content = await response.text()
                                feed = feedparser.parse(content)

                                if feed.bozo and feed.bozo_exception:
                                    logger.warning(f"Feed parsing warning for {feed_name}: {feed.bozo_exception}")

                                for entry in feed.entries:
                                    try:
                                        news_item = {
                                            'title': entry.get('title', '').strip(),
                                            'description': self._clean_html(entry.get('description', '')),
                                            'link': entry.get('link', ''),
                                            'published': self._parse_date(entry.get('published', '')),
                                            'source': feed_name,
                                            'source_url': rss_url,
                                            'type': 'rss'
                                        }

                                        # Clean and extract text
                                        news_item['text'] = self._clean_text(
                                            f"{news_item['title']} {news_item['description']}"
                                        )

                                        # Skip empty or invalid items
                                        if news_item['text'] and len(news_item['text']) > 20:
                                            news_items.append(news_item)

                                    except Exception as e:
                                        logger.debug(f"Error parsing entry from {feed_name}: {e}")
                                        continue

                                logger.info(f"Scraped {len(feed.entries)} items from {feed_name}")
                                break  # Success, exit retry loop

                            elif response.status == 429:  # Rate limited
                                wait_time = 2 ** attempt
                                logger.warning(f"Rate limited on {feed_name}. Waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.warning(f"HTTP {response.status} for {feed_name}")

                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout on {feed_name} (attempt {attempt + 1}/{retries})")
                    except aiohttp.ClientError as e:
                        logger.warning(f"Client error on {feed_name}: {e}")

                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                logger.error(f"Error scraping {feed_name}: {e}")
                continue

        logger.info(f"Total RSS news items scraped: {len(news_items)}")
        return news_items

    async def scrape_reddit_news(self, hours_back: int = 24) -> List[Dict]:
        """Scrape Reddit using JSON API (no authentication required)"""
        # Ensure session is initialized
        if self.session is None:
            logger.warning("Session not initialized, initializing now...")
            await self.initialize()
        
        news_items = []

        for subreddit in self.REDDIT_SUBREDDITS:
            try:
                await self._rate_limit_wait(f'reddit_{subreddit}')

                url = f"https://www.reddit.com/r/{subreddit}/new.json"
                params = {'limit': 50}
                headers = {
                    'User-Agent': 'Python:IndianMarketScraper:v1.0.0'
                }

                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        for post in data.get('data', {}).get('children', []):
                            try:
                                post_data = post['data']
                                post_time = datetime.fromtimestamp(post_data['created_utc'])

                                # Filter by time
                                if post_time >= datetime.now() - timedelta(hours=hours_back):
                                    score = post_data.get('score', 0)
                                    comments = post_data.get('num_comments', 0)

                                    # Quality filter
                                    min_score = 2 if subreddit in ['IndianStockMarket', 'IndianStreetBets'] else 3

                                    if score >= min_score or comments >= 3:
                                        full_text = f"{post_data['title']} {post_data.get('selftext', '')}"

                                        news_item = {
                                            'title': post_data['title'][:200],
                                            'description': post_data.get('selftext', '')[:500],
                                            'link': f"https://reddit.com{post_data['permalink']}",
                                            'published': post_time,
                                            'source': f'r/{subreddit}',
                                            'type': 'reddit',
                                            'text': self._clean_text(full_text),
                                            'score': score,
                                            'comments': comments,
                                            'upvote_ratio': post_data.get('upvote_ratio', 0)
                                        }
                                        news_items.append(news_item)

                            except Exception as e:
                                logger.debug(f"Error parsing Reddit post: {e}")
                                continue

                        logger.info(f"Scraped {len(news_items)} items from r/{subreddit}")

                    elif response.status == 429:
                        logger.warning(f"Rate limited on Reddit r/{subreddit}")
                        await asyncio.sleep(60)  # Wait 1 minute

            except Exception as e:
                logger.error(f"Error scraping r/{subreddit}: {e}")
                continue

        logger.info(f"Total Reddit news items: {len(news_items)}")
        return news_items

    async def scrape_google_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Scrape Google News RSS for specific symbol"""
        # Ensure session is initialized and not closed
        if self.session is None or self.session.closed:
            logger.warning("Session not initialized or closed, initializing now...")
            await self.initialize()
        
        news_items = []

        try:
            await self._rate_limit_wait(f'google_news_{symbol}')

            # Check session before use
            if self.session is None or self.session.closed:
                logger.warning("Session closed during Google News scraping, reinitializing...")
                await self.initialize()

            # Construct query with symbol and company name variations
            query_terms = [symbol]
            if symbol in self.symbol_patterns:
                query_terms.extend(self.symbol_patterns[symbol][:2])  # Add top 2 variations

            query = ' OR '.join(query_terms) + ' India stock'
            encoded_query = quote_plus(query)

            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)

                    for entry in feed.entries:
                        try:
                            entry_time = self._parse_date(entry.get('published', ''))

                            if entry_time >= datetime.now() - timedelta(hours=hours_back):
                                news_item = {
                                    'title': entry.get('title', '').strip(),
                                    'description': self._clean_html(entry.get('summary', '')),
                                    'link': entry.get('link', ''),
                                    'published': entry_time,
                                    'source': 'Google News',
                                    'type': 'google_news',
                                    'text': self._clean_text(f"{entry.get('title', '')} {entry.get('summary', '')}")
                                }

                                if news_item['text'] and len(news_item['text']) > 20:
                                    news_items.append(news_item)

                        except Exception as e:
                            logger.debug(f"Error parsing Google News entry: {e}")
                            continue

                    logger.info(f"Scraped {len(news_items)} Google News items for {symbol}")

        except Exception as e:
            logger.error(f"Error scraping Google News for {symbol}: {e}")

        return news_items

    async def get_recent_news(self, symbols: List[str], hours_back: int = 1) -> Dict[str, List[Dict]]:
        """Get recent news for specific symbols from all sources"""
        symbol_news = {symbol: [] for symbol in symbols}

        try:
            # Gather all news sources concurrently
            tasks = [
                self.scrape_rss_feeds(),
                self.scrape_reddit_news(hours_back),
            ]

            # Add Google News for each symbol
            for symbol in symbols:
                tasks.append(self.scrape_google_news(symbol, hours_back))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine all news
            all_news = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                elif isinstance(result, list):
                    all_news.extend(result)

            # Filter news by symbols with ML-based relevance
            for news_item in all_news:
                for symbol in symbols:
                    if self._is_news_relevant(news_item, symbol):
                        # Calculate relevance score
                        if self.relevance_model and symbol in self.symbol_embeddings:
                            news_item['relevance_score'] = self._calculate_relevance_score(news_item, symbol)
                        else:
                            news_item['relevance_score'] = 1.0

                        news_item['matched_symbol'] = symbol

                        # Avoid duplicates
                        if not any(n['link'] == news_item['link'] for n in symbol_news[symbol]):
                            symbol_news[symbol].append(news_item)

            # Sort by relevance and time
            for symbol in symbols:
                symbol_news[symbol].sort(
                    key=lambda x: (x.get('relevance_score', 0), x.get('published', datetime.min)),
                    reverse=True
                )

        except Exception as e:
            logger.error(f"Error getting recent news: {e}")

        return symbol_news

    def _calculate_relevance_score(self, news_item: Dict, symbol: str) -> float:
        """Calculate ML-based relevance score"""
        if not self.relevance_model or symbol not in self.symbol_embeddings:
            return 0.0

        try:
            news_text = f"{news_item.get('title', '')} {news_item.get('description', '')}"
            news_embedding = self.relevance_model.encode(news_text)
            symbol_embedding = self.symbol_embeddings[symbol]

            similarity = cosine_similarity([news_embedding], [symbol_embedding])[0][0]
            return float(similarity)
        except Exception as e:
            logger.debug(f"Error calculating relevance: {e}")
            return 0.0

    def _is_news_relevant(self, news_item: Dict, symbol: str, threshold: float = 0.3) -> bool:
        """ML-first relevance check with intelligent fallback"""
        text = news_item.get('text', '').lower()

        # Quick filters
        if not text or len(text) < 20:
            return False

        # Primary: Direct symbol mention (highest confidence)
        if symbol.lower() in text:
            return True

        # Exchange format
        if f"{symbol.lower()}.ns" in text or f"{symbol.lower()}.bo" in text:
            return True

        # Pattern matching
        if symbol in self.symbol_patterns:
            for pattern in self.symbol_patterns[symbol]:
                if pattern.lower() in text:
                    return True

        # ML-based scoring (if available)
        if self.relevance_model and symbol in self.symbol_embeddings:
            ml_score = self._calculate_relevance_score(news_item, symbol)
            if ml_score > threshold:
                return True

        return False

    def _clean_html(self, html: str) -> str:
        """Remove HTML tags and decode entities"""
        if not html:
            return ""

        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove HTML
        text = self._clean_html(text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s.,!?()-]', '', text)

        return text.strip()

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string with multiple format support"""
        if not date_str:
            return datetime.now()

        formats = [
            '%a, %d %b %Y %H:%M:%S %Z',
            '%a, %d %b %Y %H:%M:%S %z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%d %b %Y %H:%M:%S %Z'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Fallback
        return datetime.now()

    async def get_news_sentiment_summary(self, symbol: str, hours_back: int = 1) -> Dict:
        """Get sentiment summary for a symbol's news"""
        try:
            symbol_news = await self.get_recent_news([symbol], hours_back)
            news_items = symbol_news.get(symbol, [])

            return {
                'sentiment_score': 0.0,  # To be calculated by NLP processor
                'news_count': len(news_items),
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': len(news_items),
                'recent_news': news_items[:5],
                'sources': list(set(item.get('source', 'unknown') for item in news_items))
            }

        except Exception as e:
            logger.error(f"Error getting news sentiment summary: {e}")
            return {
                'sentiment_score': 0.0,
                'news_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'recent_news': [],
                'sources': []
            }

    async def update_symbol_list(self, new_symbols: List[str]) -> bool:
        """Dynamically update symbol list"""
        try:
            if not self.relevance_model:
                logger.warning("ML model not available for updates")
                return False

            for symbol in new_symbols:
                if symbol not in self.symbol_embeddings:
                    context = await self._create_symbol_context(symbol)
                    self.symbol_embeddings[symbol] = self.relevance_model.encode(context)
                    logger.info(f"Added symbol: {symbol}")

            return True

        except Exception as e:
            logger.error(f"Error updating symbols: {e}")
            return False


# Example usage
async def main():
    scraper = NewsScraper()
    await scraper.initialize()

    try:
        # Test with sample symbols
        symbols = ['RELIANCE', 'TCS', 'INFY']
        news = await scraper.get_recent_news(symbols, hours_back=24)

        for symbol, items in news.items():
            print(f"\n{symbol}: {len(items)} news items")
            for item in items[:3]:
                print(f"  - {item['title'][:80]}...")
                print(f"    Source: {item['source']}, Relevance: {item.get('relevance_score', 0):.2f}")

    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())