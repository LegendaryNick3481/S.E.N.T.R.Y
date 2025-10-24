"""
Live trading execution system for Mismatched Energy strategy
"""
import asyncio
import websockets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from data.fyers_client import FyersClient
from news.news_scraper import NewsScraper
from nlp.sentiment_analyzer import SentimentAnalyzer
from analysis.cross_modal_analyzer import CrossModalAnalyzer
from scoring.capital_allocator import CapitalAllocator
from utils.event_bus import event_bus

logger = logging.getLogger(__name__)

class LiveExecutor:
    def __init__(self, fyers_client: FyersClient, news_scraper: NewsScraper, sentiment_analyzer: SentimentAnalyzer):
        self.fyers_client = fyers_client
        self.news_scraper = news_scraper
        self.sentiment_analyzer = sentiment_analyzer
        self.cross_modal_analyzer = CrossModalAnalyzer()
        self.capital_allocator = CapitalAllocator()
        
        self.is_running = False
        self.watchlist = []
        self.positions = {}
        self.orders = {}
        self.performance_metrics = {}
    
    async def start_trading(self, watchlist: List[str]):
        """Start live trading with given watchlist"""
        try:
            self.watchlist = watchlist
            self.is_running = True
            
            logger.info(f"Starting live trading for {len(watchlist)} symbols: {watchlist}")
            await event_bus.publish({
                "type": "status", 
                "stage": "start", 
                "watchlist": watchlist,
                "message": f"Starting live trading for {len(watchlist)} symbols"
            })
            
            # Connect to Fyers websocket and subscribe to symbols
            logger.info("Connecting to websocket...")
            await event_bus.publish({"type": "log", "level": "INFO", "message": "Connecting to websocket..."})
            
            ws_connected = await self.fyers_client.connect_websocket()
            if not ws_connected:
                logger.error("Failed to connect to websocket")
                await event_bus.publish({"type": "error", "message": "Failed to connect to websocket"})
                return
            
            logger.info("Subscribing to symbols...")
            await event_bus.publish({"type": "log", "level": "INFO", "message": f"Subscribing to {len(watchlist)} symbols..."})
            
            subscribed = await self.fyers_client.subscribe_symbols(watchlist)
            if not subscribed:
                logger.error("Failed to subscribe to symbols")
                await event_bus.publish({"type": "error", "message": "Failed to subscribe to symbols"})
                return
            
            logger.info("Setup complete! Starting trading loop...")
            await event_bus.publish({"type": "log", "level": "SUCCESS", "message": "Setup complete! Trading loop started"})
            
            # Start main trading loop
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            await event_bus.publish({"type": "error", "message": f"Error starting trading: {e}"})
            self.is_running = False
    
    async def stop_trading(self):
        """Stop live trading"""
        try:
            self.is_running = False
            logger.info("Stopping live trading")
            
            # Close any open positions if needed
            await self._close_all_positions()
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
    
    async def _trading_loop(self):
        """Main trading loop with smart off-hours handling"""
        try:
            while self.is_running:
                # Check if market is open
                if not self._is_market_open():
                    # Get time until market opens
                    wait_seconds = self._seconds_until_market_open()
                    
                    logger.info(f"Market is closed. Next check in {wait_seconds//60} minutes")
                    await event_bus.publish({
                        "type": "status", 
                        "stage": "market_closed",
                        "next_open_seconds": wait_seconds
                    })
                    
                    # Disconnect WebSocket to save resources
                    if self.fyers_client.websocket_connected:
                        logger.info("Disconnecting WebSocket during off-hours")
                        await self.fyers_client.disconnect_websocket()
                    
                    # Smart wait: longer during off-hours
                    await asyncio.sleep(min(wait_seconds, 300))  # Max 5 min check
                    continue
                
                # Market is open - ensure WebSocket is connected
                if not self.fyers_client.websocket_connected:
                    logger.info("Market opened - reconnecting WebSocket")
                    await self.fyers_client.connect_websocket()
                    await self.fyers_client.subscribe_symbols(self.watchlist)
                
                # Process trading cycle
                await self._process_trading_cycle()
                
                # Wait before next cycle
                await asyncio.sleep(Config.LOOKBACK_PERIODS['price'])  # 5 seconds
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            self.is_running = False
    
    def _seconds_until_market_open(self) -> int:
        """Calculate seconds until market opens"""
        try:
            import pytz
            
            ist = pytz.timezone(Config.MARKET_HOURS['timezone'])
            now_ist = datetime.now(ist)
            
            # If weekend, wait until Monday
            if now_ist.weekday() == 5:  # Saturday
                days_until_monday = 2
            elif now_ist.weekday() == 6:  # Sunday
                days_until_monday = 1
            else:
                days_until_monday = 0
            
            # Calculate next market open time
            if days_until_monday > 0:
                # Next Monday at market open
                next_open = now_ist + timedelta(days=days_until_monday)
                next_open = next_open.replace(hour=9, minute=15, second=0, microsecond=0)
            else:
                # Check if today's market hasn't opened yet
                market_open_today = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
                if now_ist < market_open_today:
                    next_open = market_open_today
                else:
                    # Tomorrow's market open
                    next_open = (now_ist + timedelta(days=1)).replace(hour=9, minute=15, second=0, microsecond=0)
            
            seconds = int((next_open - now_ist).total_seconds())
            return max(60, seconds)  # At least 60 seconds
            
        except Exception as e:
            logger.error(f"Error calculating next market open: {e}")
            return 300  # Default 5 minutes
    
    async def _process_trading_cycle(self):
        """Process one trading cycle"""
        try:
            logger.info("Processing trading cycle...")
            await event_bus.publish({"type": "cycle", "stage": "begin", "timestamp": datetime.now().isoformat()})
            
            # 1. Detect significant price moves
            significant_moves = await self.fyers_client.detect_significant_moves(self.watchlist)
            await event_bus.publish({"type": "moves", "data": significant_moves})
            
            if not significant_moves:
                logger.info("No significant moves detected")
                return
            
            logger.info(f"Detected {len(significant_moves)} significant moves")
            
            # 2. Get news for moving symbols
            moving_symbols = [symbol for symbol, _ in significant_moves]
            news_data = await self.news_scraper.get_recent_news(moving_symbols, hours_back=1)
            await event_bus.publish({"type": "news", "symbols": moving_symbols, "counts": {s: len(news_data.get(s, [])) for s in moving_symbols}})
            
            # 3. Analyze each symbol
            candidate_symbols = []
            
            for symbol, price_change in significant_moves:
                try:
                    # Get current price and volume data
                    microstructure = await self.fyers_client.get_market_microstructure(symbol)
                    price_features = await self.fyers_client.calculate_price_features(symbol)
                    
                    # Log price information
                    current_price = microstructure.get('last_price', 0)
                    volume = microstructure.get('volume', 0)
                    logger.info(f"{symbol}: ₹{current_price:.2f} | Change: {price_change:+.2f}% | Volume: {volume:,}")
                    
                    # Analyze news sentiment
                    symbol_news = news_data.get(symbol, [])
                    analyzed_news = self.sentiment_analyzer.analyze_news_batch(symbol_news)
                    sentiment_summary = self.sentiment_analyzer.calculate_news_sentiment_summary(analyzed_news)
                    await event_bus.publish({"type": "sentiment", "symbol": symbol, "summary": sentiment_summary})
                    
                    # Publish individual news items to dashboard
                    for news_item in analyzed_news[:3]:  # Top 3 news items
                        await event_bus.publish({
                            "type": "news_item",
                            "data": {
                                "symbol": symbol,
                                "title": news_item.get('title', 'No title'),
                                "relevance_score": news_item.get('relevance_score', 0),
                                "sentiment": news_item.get('sentiment', 0)
                            }
                        })
                    
                    # Calculate cross-modal analysis
                    news_embeddings = np.array([item['embedding'] for item in analyzed_news])
                    if news_embeddings.size > 0:
                        news_embedding = np.mean(news_embeddings, axis=0)
                    else:
                        news_embedding = np.zeros(384)
                    
                    # Detect mismatched energy
                    mismatch_analysis = self.cross_modal_analyzer.detect_mismatched_energy(
                        sentiment_summary['weighted_sentiment'],
                        price_change,
                        microstructure.get('volume', 0),
                        microstructure.get('volatility', 0)
                    )
                    await event_bus.publish({"type": "mismatch", "symbol": symbol, "analysis": mismatch_analysis})
                    
                    if mismatch_analysis['is_mismatched']:
                        logger.info(f"MISMATCH DETECTED: {symbol} | Discord Score: {mismatch_analysis['discord_score']:.3f} | Confidence: {mismatch_analysis['confidence']:.1%}")
                        candidate_symbols.append({
                            'symbol': symbol,
                            'discord_score': mismatch_analysis['discord_score'],
                            'confidence': mismatch_analysis['confidence'],
                            'price_change': price_change,
                            'current_price': microstructure.get('last_price', 0),
                            'volume_anomaly': microstructure.get('volume', 0),
                            'volatility_spike': microstructure.get('volatility', 0),
                            'sentiment_summary': sentiment_summary
                        })
                        
                except Exception as e:
                    logger.error(f"Error analyzing symbol {symbol}: {e}")
                    continue
            
            if not candidate_symbols:
                logger.info("No mismatched energy opportunities found")
                return
            
            # 4. Optimize portfolio allocation
            allocations = self.capital_allocator.optimize_portfolio_allocation(candidate_symbols)
            await event_bus.publish({"type": "allocations", "data": allocations})
            
            if not allocations:
                logger.info("No valid allocations found")
                return
            
            # 5. Generate and execute trading signals
            signals = self.capital_allocator.generate_trading_signals(allocations)
            await event_bus.publish({"type": "signals", "data": signals})
            
            if signals:
                await self._execute_signals(signals)
            
            # 6. Update performance metrics
            await self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error processing trading cycle: {e}")
            await event_bus.publish({"type": "error", "message": str(e)})
    
    async def _execute_signals(self, signals: List[Dict]):
        """Execute trading signals"""
        try:
            for signal in signals:
                symbol = signal['symbol']
                action = signal['action']
                quantity = signal['quantity']
                price = signal.get('price', 0)
                
                print(f"📊 SIGNAL: {action.upper()} {quantity} shares of {symbol} @ ₹{price:.2f}")
                logger.info(f"Executing {action} signal for {symbol}: {quantity} shares")
                
                # Place order
                order_result = await self.fyers_client.place_order(
                    symbol=symbol,
                    side=action,
                    quantity=quantity,
                    order_type="MARKET"
                )
                
                if order_result.get('code') == 200:
                    # Order placed successfully
                    order_id = order_result.get('id')
                    self.orders[order_id] = {
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'timestamp': datetime.now(),
                        'status': 'pending'
                    }
                    
                    logger.info(f"Order placed successfully: {order_id}")
                else:
                    logger.error(f"Failed to place order: {order_result}")
                    
        except Exception as e:
            logger.error(f"Error executing signals: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Get current positions
            positions = await self.fyers_client.get_positions()
            self.positions = {pos['symbol']: pos for pos in positions}
            
            # Get account summary
            account_summary = await self.fyers_client.get_account_summary()
            
            # Calculate portfolio value
            portfolio_value = account_summary.get('equity_amount', 0)
            
            # Update capital allocator
            self.capital_allocator.update_portfolio_value(portfolio_value)
            
            # Calculate performance metrics
            performance = self.capital_allocator.calculate_portfolio_performance()
            
            self.performance_metrics = {
                'portfolio_value': portfolio_value,
                'positions': self.positions,
                'performance': performance,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Portfolio value: ₹{portfolio_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            positions = await self.fyers_client.get_positions()
            
            for position in positions:
                symbol = position['symbol']
                quantity = position['net_quantity']
                
                if quantity > 0:
                    # Close long position
                    await self.fyers_client.place_order(
                        symbol=symbol,
                        side="SELL",
                        quantity=quantity,
                        order_type="MARKET"
                    )
                    logger.info(f"Closed long position in {symbol}: {quantity} shares")
                
                elif quantity < 0:
                    # Close short position
                    await self.fyers_client.place_order(
                        symbol=symbol,
                        side="BUY",
                        quantity=abs(quantity),
                        order_type="MARKET"
                    )
                    logger.info(f"Closed short position in {symbol}: {abs(quantity)} shares")
                    
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def _is_market_open(self) -> bool:
        """Check if market is open (with timezone and weekend/holiday support)"""
        try:
            import pytz
            
            # Get current time in IST
            ist = pytz.timezone(Config.MARKET_HOURS['timezone'])
            now_ist = datetime.now(ist)
            
            # Check if weekend (Saturday=5, Sunday=6)
            if now_ist.weekday() in [5, 6]:
                return False
            
            # Check trading hours
            current_time = now_ist.strftime('%H:%M')
            market_start = Config.MARKET_HOURS['start']
            market_end = Config.MARKET_HOURS['end']
            
            is_open = market_start <= current_time <= market_end
            
            if not is_open:
                # Log next market open time
                if current_time < market_start:
                    logger.debug(f"Market opens at {market_start} IST")
                elif current_time > market_end:
                    logger.debug(f"Market closed. Opens tomorrow at {market_start} IST")
            
            return is_open
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    async def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        try:
            return {
                'is_running': self.is_running,
                'watchlist': self.watchlist,
                'positions': self.positions,
                'orders': self.orders,
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {}
    
    async def add_to_watchlist(self, symbols: List[str]):
        """Add symbols to watchlist"""
        try:
            for symbol in symbols:
                if symbol not in self.watchlist:
                    self.watchlist.append(symbol)
            
            logger.info(f"Added {len(symbols)} symbols to watchlist")
            
        except Exception as e:
            logger.error(f"Error adding to watchlist: {e}")
    
    async def remove_from_watchlist(self, symbols: List[str]):
        """Remove symbols from watchlist"""
        try:
            for symbol in symbols:
                if symbol in self.watchlist:
                    self.watchlist.remove(symbol)
            
            logger.info(f"Removed {len(symbols)} symbols from watchlist")
            
        except Exception as e:
            logger.error(f"Error removing from watchlist: {e}")
    
    async def get_trading_signals(self) -> List[Dict]:
        """Get current trading signals"""
        try:
            if not self.watchlist:
                return []
            
            # Get recent news and price data
            news_data = await self.news_scraper.get_recent_news(self.watchlist, hours_back=1)
            
            signals = []
            for symbol in self.watchlist:
                try:
                    # Get current price from websocket
                    current_price = await self.fyers_client.get_latest_price(symbol)
                    if current_price is None:
                        logger.warning(f"No live price data for {symbol} from websocket.")
                        continue
                    
                    # Analyze news
                    symbol_news = news_data.get(symbol, [])
                    analyzed_news = self.sentiment_analyzer.analyze_news_batch(symbol_news)
                    sentiment_summary = self.sentiment_analyzer.calculate_news_sentiment_summary(analyzed_news)
                    
                    # Calculate price features
                    price_features = await self.fyers_client.calculate_price_features(symbol)
                    
                    # Cross-modal analysis
                    news_embeddings = np.array([item['embedding'] for item in analyzed_news])
                    if news_embeddings.size > 0:
                        news_embedding = np.mean(news_embeddings, axis=0)
                    else:
                        news_embedding = np.zeros(384)
                    
                    correlation_analysis = self.cross_modal_analyzer.analyze_cross_modal_correlation(
                        news_embedding, price_features
                    )
                    
                    if correlation_analysis['discord_score'] > Config.MIN_CORRELATION_THRESHOLD:
                        signals.append({
                            'symbol': symbol,
                            'discord_score': correlation_analysis['discord_score'],
                            'confidence': correlation_analysis['confidence'],
                            'sentiment': sentiment_summary['weighted_sentiment'],
                            'current_price': current_price,
                            'timestamp': datetime.now()
                        })
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting trading signals: {e}")
            return []

