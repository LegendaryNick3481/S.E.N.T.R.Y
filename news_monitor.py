"""
Terminal-based news monitoring system
Displays all scraped news in a clean, organized format
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List
import sys
from colorama import init, Fore, Back, Style
from news.news_scraper import NewsScraper

# Initialize colorama for Windows color support
init(autoreset=True)

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('news_monitor.log')]
)
logger = logging.getLogger(__name__)


class NewsMonitor:
    def __init__(self):
        self.scraper = NewsScraper()
        self.is_running = False
        
    async def initialize(self):
        """Initialize the news scraper"""
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}{'NEWS MONITOR - Initializing...':^80}")
        print(f"{Fore.CYAN}{'='*80}\n")
        
        await self.scraper.initialize()
        print(f"{Fore.GREEN}✓ News scraper initialized\n")
        
    async def display_news_once(self, symbols: List[str], hours_back: int = 1):
        """Fetch and display news once"""
        self._clear_screen()
        
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}{f'NEWS MONITOR - {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}':^80}")
        print(f"{Fore.CYAN}{'='*80}\n")
        
        print(f"{Fore.YELLOW}📡 Fetching news for {len(symbols)} symbols...")
        print(f"{Fore.YELLOW}   Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}\n")
        
        # Get news
        news_data = await self.scraper.get_recent_news(symbols, hours_back=hours_back)
        
        # Display by symbol
        total_news = 0
        for symbol in symbols:
            news_items = news_data.get(symbol, [])
            if news_items:
                total_news += len(news_items)
                self._display_symbol_news(symbol, news_items)
        
        # Summary
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}✓ Total: {total_news} news items for {len([s for s in symbols if news_data.get(s)])} symbols")
        print(f"{Fore.CYAN}{'='*80}\n")
        
    async def monitor_continuous(self, symbols: List[str], hours_back: int = 1, refresh_minutes: int = 5):
        """Continuously monitor and display news"""
        self.is_running = True
        
        try:
            while self.is_running:
                await self.display_news_once(symbols, hours_back)
                
                # Countdown to next refresh
                print(f"{Fore.YELLOW}Next refresh in {refresh_minutes} minutes. Press Ctrl+C to exit.")
                
                for remaining in range(refresh_minutes * 60, 0, -10):
                    mins, secs = divmod(remaining, 60)
                    sys.stdout.write(f"\r{Fore.YELLOW}⏱️  Refreshing in {mins:02d}:{secs:02d}...  ")
                    sys.stdout.flush()
                    await asyncio.sleep(10)
                
                sys.stdout.write("\r" + " " * 50 + "\r")
                sys.stdout.flush()
                
        except KeyboardInterrupt:
            print(f"\n\n{Fore.RED}🛑 Monitoring stopped by user")
        finally:
            self.is_running = False
            await self.scraper.close()
    
    def _display_symbol_news(self, symbol: str, news_items: List[Dict]):
        """Display news for a single symbol"""
        print(f"\n{Fore.MAGENTA}{'─'*80}")
        print(f"{Fore.MAGENTA}📊 {symbol} ({len(news_items)} items)")
        print(f"{Fore.MAGENTA}{'─'*80}")
        
        for idx, item in enumerate(news_items[:10], 1):  # Show top 10
            # Header with source and time
            source = item.get('source', 'Unknown')
            published = item.get('published', datetime.now())
            if isinstance(published, datetime):
                time_str = published.strftime('%H:%M')
            else:
                time_str = 'N/A'
            
            relevance = item.get('relevance_score', 0)
            
            print(f"\n{Fore.CYAN}{idx}. [{source}] {time_str} | Relevance: {relevance:.2f}")
            
            # Title
            title = item.get('title', 'No title')
            print(f"{Fore.WHITE}{Style.BRIGHT}{title}")
            
            # Description (first 200 chars)
            description = item.get('description', '')
            if description:
                desc_short = description[:200] + '...' if len(description) > 200 else description
                print(f"{Fore.WHITE}{desc_short}")
            
            # Link
            link = item.get('link', '')
            if link:
                print(f"{Fore.BLUE}{Style.DIM}{link}")
            
            # Reddit-specific data
            if item.get('type') == 'reddit':
                score = item.get('score', 0)
                comments = item.get('comments', 0)
                print(f"{Fore.GREEN}👍 {score} | 💬 {comments} comments")
        
        if len(news_items) > 10:
            print(f"\n{Fore.YELLOW}... and {len(news_items) - 10} more items")
    
    def _clear_screen(self):
        """Clear terminal screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    async def display_all_sources(self, hours_back: int = 24):
        """Display news from all sources without filtering by symbol"""
        self._clear_screen()
        
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}{f'ALL NEWS SOURCES - {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}':^80}")
        print(f"{Fore.CYAN}{'='*80}\n")
        
        # Scrape all RSS feeds
        print(f"{Fore.YELLOW}📰 Scraping RSS feeds...")
        rss_news = await self.scraper.scrape_rss_feeds()
        print(f"{Fore.GREEN}✓ {len(rss_news)} items from RSS\n")
        
        # Scrape Reddit
        print(f"{Fore.YELLOW}🔴 Scraping Reddit...")
        reddit_news = await self.scraper.scrape_reddit_news(hours_back)
        print(f"{Fore.GREEN}✓ {len(reddit_news)} items from Reddit\n")
        
        # Display RSS news
        if rss_news:
            print(f"\n{Fore.MAGENTA}{'='*80}")
            print(f"{Fore.MAGENTA}{'RSS FEEDS':^80}")
            print(f"{Fore.MAGENTA}{'='*80}")
            
            # Group by source
            sources = {}
            for item in rss_news:
                source = item.get('source', 'Unknown')
                sources.setdefault(source, []).append(item)
            
            for source, items in sources.items():
                print(f"\n{Fore.CYAN}📰 {source} ({len(items)} items)")
                print(f"{Fore.CYAN}{'─'*80}")
                
                for idx, item in enumerate(items[:5], 1):
                    self._display_news_item(item, idx)
                
                if len(items) > 5:
                    print(f"{Fore.YELLOW}... and {len(items) - 5} more items")
        
        # Display Reddit news
        if reddit_news:
            print(f"\n{Fore.MAGENTA}{'='*80}")
            print(f"{Fore.MAGENTA}{'REDDIT':^80}")
            print(f"{Fore.MAGENTA}{'='*80}")
            
            # Group by subreddit
            subreddits = {}
            for item in reddit_news:
                source = item.get('source', 'Unknown')
                subreddits.setdefault(source, []).append(item)
            
            for subreddit, items in subreddits.items():
                print(f"\n{Fore.CYAN}🔴 {subreddit} ({len(items)} items)")
                print(f"{Fore.CYAN}{'─'*80}")
                
                for idx, item in enumerate(items[:5], 1):
                    self._display_news_item(item, idx, show_reddit_stats=True)
                
                if len(items) > 5:
                    print(f"{Fore.YELLOW}... and {len(items) - 5} more items")
        
        # Summary
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}✓ Total: {len(rss_news) + len(reddit_news)} news items")
        print(f"{Fore.CYAN}{'='*80}\n")
    
    def _display_news_item(self, item: Dict, idx: int, show_reddit_stats: bool = False):
        """Display a single news item"""
        published = item.get('published', datetime.now())
        if isinstance(published, datetime):
            time_str = published.strftime('%Y-%m-%d %H:%M')
        else:
            time_str = 'N/A'
        
        print(f"\n{Fore.CYAN}{idx}. [{time_str}]")
        
        # Title
        title = item.get('title', 'No title')
        print(f"{Fore.WHITE}{Style.BRIGHT}{title}")
        
        # Description (first 150 chars)
        description = item.get('description', '')
        if description:
            desc_short = description[:150] + '...' if len(description) > 150 else description
            print(f"{Fore.WHITE}{desc_short}")
        
        # Reddit stats
        if show_reddit_stats:
            score = item.get('score', 0)
            comments = item.get('comments', 0)
            ratio = item.get('upvote_ratio', 0)
            print(f"{Fore.GREEN}👍 {score} | 💬 {comments} | 📊 {ratio:.0%} upvoted")
        
        # Link
        link = item.get('link', '')
        if link:
            print(f"{Fore.BLUE}{Style.DIM}{link}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='News Monitor - Terminal-based news display')
    parser.add_argument('--mode', choices=['symbols', 'all'], default='symbols',
                       help='Display mode: symbols (filtered by symbols) or all (all sources)')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Symbols to monitor (default: loads from tickers)')
    parser.add_argument('--hours', type=int, default=1,
                       help='Hours of news to look back (default: 1)')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuously refresh news')
    parser.add_argument('--refresh', type=int, default=5,
                       help='Refresh interval in minutes for continuous mode (default: 5)')
    
    args = parser.parse_args()
    
    # Load symbols if not provided
    if args.symbols is None and args.mode == 'symbols':
        try:
            from data.tickers import get_tickers
            args.symbols = get_tickers()
            print(f"Loaded {len(args.symbols)} symbols from tickers")
        except Exception as e:
            print(f"Error loading tickers: {e}")
            args.symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    
    # Create monitor
    monitor = NewsMonitor()
    
    try:
        await monitor.initialize()
        
        if args.mode == 'all':
            if args.continuous:
                print(f"{Fore.YELLOW}Continuous mode not available for 'all' sources mode")
                await monitor.display_all_sources(args.hours)
            else:
                await monitor.display_all_sources(args.hours)
        else:
            if args.continuous:
                await monitor.monitor_continuous(args.symbols, args.hours, args.refresh)
            else:
                await monitor.display_news_once(args.symbols, args.hours)
        
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Exiting...")
    finally:
        await monitor.scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
