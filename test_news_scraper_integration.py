"""
Test script to verify news_scraper integration
"""
import asyncio
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_news_scraper():
    """Test the news scraper integration"""
    try:
        logger.info("Testing news scraper import...")
        from news.news_scraper import NewsScraper
        logger.info("✓ Import successful")
        
        logger.info("Testing initialization...")
        scraper = NewsScraper()
        await scraper.initialize()
        logger.info("✓ Initialization successful")
        
        logger.info("Testing ticker import...")
        from data.tickers import get_tickers
        symbols = get_tickers()
        logger.info(f"✓ Loaded {len(symbols)} symbols: {symbols}")
        
        logger.info("Testing news fetching for a single symbol...")
        news_data = await scraper.get_recent_news([symbols[0]], hours_back=1)
        logger.info(f"✓ Fetched news for {symbols[0]}: {len(news_data.get(symbols[0], []))} items")
        
        logger.info("Testing symbol update...")
        success = await scraper.update_symbol_list(['TEST', 'DEMO'])
        logger.info(f"✓ Symbol update: {'Success' if success else 'Failed (expected if no ML model)'}")
        
        await scraper.close()
        logger.info("✓ Cleanup successful")
        
        logger.info("\n=== ALL TESTS PASSED ===")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    result = asyncio.run(test_news_scraper())
    sys.exit(0 if result else 1)
