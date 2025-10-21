"""
Main orchestration system for Mismatched Energy trading strategy
The central hub that coordinates all components
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import json

# Import all components
from data.fyers_client import FyersClient
from news.news_scraper import NewsScraper
from nlp.sentiment_analyzer import SentimentAnalyzer
from analysis.cross_modal_analyzer import CrossModalAnalyzer
from scoring.capital_allocator import CapitalAllocator
from trading.live_executor import LiveExecutor
from backtesting.backtest_engine import BacktestEngine
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# Suppress noisy external library logs
logging.getLogger('snscrape').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class MismatchedEnergySystem:
    def __init__(self):
        self.fyers_client = FyersClient()
        self.news_scraper = NewsScraper()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.cross_modal_analyzer = CrossModalAnalyzer()
        self.capital_allocator = CapitalAllocator()
        self.live_executor = LiveExecutor()
        self.backtest_engine = BacktestEngine()
        
        self.is_running = False
        self.mode = 'backtest'  # 'backtest' or 'live'
        
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Mismatched Energy System...")
            
            # Initialize Fyers client
            if not await self.fyers_client.initialize():
                raise Exception("Failed to initialize Fyers client")
            
            # Initialize news scraper
            await self.news_scraper.initialize()
            
            # Initialize sentiment analyzer
            if not await self.sentiment_analyzer.initialize():
                raise Exception("Failed to initialize sentiment analyzer")
            
            # Initialize live executor
            if not await self.live_executor.initialize():
                raise Exception("Failed to initialize live executor")
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False
    
    async def run_backtest(self, symbols: List[str], start_date: str, end_date: str):
        """Run backtest for the strategy"""
        try:
            logger.info(f"Starting backtest for symbols: {symbols}")
            logger.info(f"Period: {start_date} to {end_date}")
            
            # Run backtest
            results = await self.backtest_engine.run_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            if results:
                # Generate report
                report = self.backtest_engine.generate_report(results)
                print(report)
                
                # Plot results
                self.backtest_engine.plot_results(results, 'backtest_results.png')
                
                # Save results to file
                with open('backtest_results.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                logger.info("Backtest completed successfully")
                return results
            else:
                logger.error("Backtest failed")
                return None
                
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    async def run_live_trading(self, watchlist: List[str]):
        """Run live trading with given watchlist"""
        try:
            logger.info(f"Starting live trading for symbols: {watchlist}")
            
            # Start live trading
            await self.live_executor.start_trading(watchlist)
            
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
    
    async def analyze_single_symbol(self, symbol: str) -> Dict:
        """Analyze a single symbol for mismatched energy"""
        try:
            logger.info(f"Analyzing symbol: {symbol}")
            
            # Get recent news
            news_data = await self.news_scraper.get_recent_news([symbol], hours_back=1)
            symbol_news = news_data.get(symbol, [])
            
            # Analyze sentiment
            analyzed_news = self.sentiment_analyzer.analyze_news_batch(symbol_news)
            sentiment_summary = self.sentiment_analyzer.calculate_news_sentiment_summary(analyzed_news)
            
            # Get price data
            price_features = await self.fyers_client.calculate_price_features(symbol)
            microstructure = await self.fyers_client.get_market_microstructure(symbol)
            
            # Calculate cross-modal analysis
            news_embeddings = np.array([item['embedding'] for item in analyzed_news])
            if news_embeddings.size > 0:
                news_embedding = np.mean(news_embeddings, axis=0)
            else:
                news_embedding = np.zeros(384)
            
            correlation_analysis = self.cross_modal_analyzer.analyze_cross_modal_correlation(
                news_embedding, price_features
            )
            
            # Detect mismatched energy
            price_change = microstructure.get('price_change', 0.0)
            volume_anomaly = self.cross_modal_analyzer.calculate_volume_anomaly(
                microstructure.get('volume', 0),
                [microstructure.get('volume', 0)]  # Simplified historical data
            )
            volatility_spike = self.cross_modal_analyzer.calculate_volatility_spike(
                microstructure.get('volatility', 0),
                [microstructure.get('volatility', 0)]  # Simplified historical data
            )
            
            mismatch_analysis = self.cross_modal_analyzer.detect_mismatched_energy(
                sentiment_summary['weighted_sentiment'],
                price_change,
                volume_anomaly,
                volatility_spike
            )
            
            return {
                'symbol': symbol,
                'sentiment_summary': sentiment_summary,
                'correlation_analysis': correlation_analysis,
                'mismatch_analysis': mismatch_analysis,
                'microstructure': microstructure,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return {}
    
    async def get_market_overview(self) -> Dict:
        """Get market overview with all symbols"""
        try:
            # Load symbols from tickers
            try:
                from data.tickers import get_tickers
                symbols = get_tickers()
            except Exception as e:
                symbols = ['HINDZINC', 'MANKIND', 'INDUSTOWER', 'DEEPINDS', 'FMGOETZE']
            
            overview = {
                'timestamp': datetime.now(),
                'symbols': {},
                'market_summary': {}
            }
            
            for symbol in symbols:
                try:
                    analysis = await self.analyze_single_symbol(symbol)
                    overview['symbols'][symbol] = analysis
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Calculate market summary
            discord_scores = [
                data.get('mismatch_analysis', {}).get('discord_score', 0)
                for data in overview['symbols'].values()
            ]
            
            overview['market_summary'] = {
                'total_symbols': len(overview['symbols']),
                'avg_discord_score': np.mean(discord_scores) if discord_scores else 0,
                'max_discord_score': np.max(discord_scores) if discord_scores else 0,
                'mismatched_symbols': len([s for s in discord_scores if s > 0.3])
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        try:
            logger.info("Shutting down Mismatched Energy System...")
            
            # Stop live trading if running
            if self.live_executor.is_running:
                await self.live_executor.stop_trading()
            
            # Close news scraper
            await self.news_scraper.close()
            
            self.is_running = False
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Mismatched Energy Trading System')
    parser.add_argument('--mode', choices=['backtest', 'live', 'analyze'], 
                       default='backtest', help='Operation mode')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Symbols to analyze/trade (if not provided, loads from data/tickers.py)')
    parser.add_argument('--start-date', default='2023-01-01', 
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-01-01',
                       help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--watchlist', nargs='+', default=None,
                       help='Watchlist for live trading (if not provided, loads from data/tickers.py)')
    
    args = parser.parse_args()
    
    # Load symbols from tickers if not provided
    if args.symbols is None:
        try:
            from data.tickers import get_tickers
            args.symbols = get_tickers()
            logger.info(f"Loaded {len(args.symbols)} symbols: {args.symbols}")
        except Exception as e:
            logger.error(f"Could not load symbols from tickers: {e}")
            args.symbols = ['HINDZINC', 'MANKIND', 'INDUSTOWER', 'DEEPINDS', 'FMGOETZE']
    
    if args.watchlist is None:
        args.watchlist = args.symbols
    
    # Create system instance
    system = MismatchedEnergySystem()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(system.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        if not await system.initialize():
            logger.error("Failed to initialize system")
            return
        
        # Run based on mode
        if args.mode == 'backtest':
            logger.info("Running backtest mode")
            await system.run_backtest(args.symbols, args.start_date, args.end_date)
            
        elif args.mode == 'live':
            logger.info("Running live trading mode")
            await system.run_live_trading(args.watchlist)
            
        elif args.mode == 'analyze':
            logger.info("Running analysis mode")
            for symbol in args.symbols:
                analysis = await system.analyze_single_symbol(symbol)
                print(f"\nAnalysis for {symbol}:")
                print(json.dumps(analysis, indent=2, default=str))
            
            # Get market overview
            overview = await system.get_market_overview()
            print(f"\nMarket Overview:")
            print(json.dumps(overview, indent=2, default=str))
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    # Import numpy here to avoid issues
    import numpy as np
    
    # Run the main function
    asyncio.run(main())

