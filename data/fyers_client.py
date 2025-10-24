"""
Fyers API client for real-time market data and trading
"""
from fyers_apiv3 import fyersModel
import asyncio
import json
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket.data_ws import FyersDataSocket
from config import Config
import logging

logger = logging.getLogger(__name__)

class FyersClient:
    def __init__(self):
        self.fyers = None
        self.data_ws = None
        self.is_connected = False
        self.websocket_connected = False
        self.subscribed_symbols = set()
        self.price_data = {}
        self.volume_data = {}
        self.previous_close_cache = {}
        self.ws_connected_event = asyncio.Event()
        
        # Setup websocket message logger to separate file
        self.ws_logger = logging.getLogger('websocket_feed')
        self.ws_logger.setLevel(logging.INFO)
        ws_handler = logging.FileHandler('logs/websocket_feed.log', mode='a')
        ws_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.ws_logger.addHandler(ws_handler)
        self.ws_logger.propagate = False  # Don't propagate to root logger
        
    async def initialize(self):
        """Initialize Fyers API connection"""
        try:
            self.fyers = fyersModel.FyersModel(
                client_id=Config.FYERS_APP_ID,
                token=Config.FYERS_ACCESS_TOKEN,
                log_path="logs/",
                is_async=True
            )
            
            # Test connection
            profile = await self.fyers.get_profile()
            if profile['code'] == 200:
                logger.info("Fyers API connection established")
                self.is_connected = True
                
                # Initialize FyersDataSocket
                self.data_ws = FyersDataSocket(
                    access_token=f'{Config.FYERS_APP_ID}:{Config.FYERS_ACCESS_TOKEN}',
                    log_path="logs/",
                    litemode=False,
                    write_to_file=False,
                    reconnect=True,
                    on_connect=self._on_open,
                    on_close=self._on_close,
                    on_error=self._on_error,
                    on_message=self._on_message,
                    reconnect_retry=50
                )
                
                return True
            else:
                logger.error(f"Failed to connect to Fyers API: {profile}")
                self.is_connected = False
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Fyers client: {e}")
            self.is_connected = False
            return False
    
    async def get_historical_data(self, symbol: str, timeframe: str = "1", 
                                period: int = 1) -> pd.DataFrame:
        """Get historical OHLC data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period)
            
            from data.tickers import get_fyers_symbol
            fyers_symbol = get_fyers_symbol(symbol)

            data = {
                "symbol": fyers_symbol,
                "resolution": timeframe,
                "date_format": "1",
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": end_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            
            response = await self.fyers.history(data)
            
            if response['code'] == 200:
                df = pd.DataFrame(response['candles'])
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                logger.error(f"Failed to get historical data: {response}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    async def get_live_quotes(self, symbols: List[str]) -> Dict:
        """Get live quotes for symbols"""
        try:
            if not self.fyers:
                await self.initialize()
            
            # Convert symbols to Fyers format
            from data.tickers import get_fyers_symbol
            fyers_symbols = [get_fyers_symbol(symbol) for symbol in symbols]
            
            data = {
                "symbols": ",".join(fyers_symbols)
            }
            
            response = await self.fyers.quotes(data)
            
            if response['code'] == 200:
                return response['d']
            else:
                logger.error(f"Failed to get live quotes: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting live quotes: {e}")
            return {}

    def _on_message(self, message):
        # Process incoming websocket messages
        # Log to separate websocket feed file only
        self.ws_logger.info(json.dumps(message) if isinstance(message, dict) else str(message))
        
        # Skip connection/config messages
        if isinstance(message, dict):
            msg_type = message.get('type')
            if msg_type in ['cn', 'ful', 'litemode', 'sf']:
                return
        
        # Process market data messages - Exact format from StockBot.py
        if isinstance(message, dict):
            symbol = message.get('symbol')
            
            if not symbol:
                # Unknown message format - only log to file
                if message.get('type') not in ['cn', 'ful', 'litemode', 'sf']:
                    logger.debug(f"Message without symbol: {json.dumps(message)[:200]}")
                return
            
            # Extract price data - exact field names from StockBot.py
            ltp = message.get('ltp')  # Last traded price
            
            if ltp is None:
                logger.debug(f"Message for {symbol} without ltp: {json.dumps(message)[:200]}")
                return
            
            # Store the price
            self.price_data[symbol] = ltp
            
            # Extract additional fields available in Fyers messages
            volume = message.get('volume', 0)
            low_price = message.get('low_price', 0)
            high_price = message.get('high_price', 0)
            open_price = message.get('open_price', 0)
            prev_close = message.get('prev_close_price', 0)
            last_traded_time = message.get('last_traded_time') or message.get('exch_feed_time')
            
            # Store volume data
            if volume:
                self.volume_data[symbol] = volume
            
            # Calculate change %
            if prev_close and prev_close > 0:
                change_pct = ((ltp - prev_close) / prev_close) * 100
            else:
                change_pct = 0
            
            # Format timestamp if available
            time_str = ""
            if last_traded_time:
                try:
                    if isinstance(last_traded_time, str):
                        last_traded_time = float(last_traded_time)
                    dt = datetime.fromtimestamp(last_traded_time)
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    pass
            
            # Publish price update to event bus for dashboard
            try:
                from utils.event_bus import event_bus
                asyncio.create_task(event_bus.publish({
                    "type": "price_update",
                    "symbol": symbol.replace('NSE:', '').replace('-EQ', ''),
                    "price": ltp,
                    "change": change_pct,
                    "volume": volume,
                    "high": high_price,
                    "low": low_price,
                    "time": time_str
                }))
            except:
                pass
            
            logger.debug(f"Live price update for {symbol}: {ltp}")

    def _on_open(self):
        logger.info("Fyers websocket connection opened.")
        self.ws_logger.info("=== WEBSOCKET CONNECTED ===")
        self.websocket_connected = True
        self.ws_connected_event.set()
        
        # Publish websocket status to event bus
        try:
            from utils.event_bus import event_bus
            asyncio.create_task(event_bus.publish({
                "type": "websocket_status",
                "connected": True
            }))
            asyncio.create_task(event_bus.publish({
                "type": "log",
                "level": "SUCCESS",
                "message": "Websocket connected"
            }))
        except:
            pass

    def _on_close(self):
        logger.info("Fyers websocket connection closed.")
        self.ws_logger.info("=== WEBSOCKET CLOSED ===")
        self.websocket_connected = False
        self.ws_connected_event.clear()
        
        # Publish websocket status to event bus
        try:
            from utils.event_bus import event_bus
            asyncio.create_task(event_bus.publish({
                "type": "websocket_status",
                "connected": False
            }))
            asyncio.create_task(event_bus.publish({
                "type": "log",
                "level": "WARNING",
                "message": "Websocket disconnected"
            }))
        except:
            pass

    def _on_error(self, message):
        logger.error(f"Fyers websocket error: {message}")
        self.ws_logger.error(f"ERROR: {message}")

    async def connect_websocket(self):
        if not self.data_ws:
            logger.info("Websocket not initialized, initializing now...")
            await self.initialize()
        if self.data_ws and not self.websocket_connected:
            logger.info("Connecting to Fyers websocket...")
            self.data_ws.connect()
            
            # Wait for connection with timeout
            try:
                await asyncio.wait_for(self.ws_connected_event.wait(), timeout=10)
                logger.info("Fyers websocket connected.")
                return True
            except asyncio.TimeoutError:
                logger.error("Websocket connection timeout")
                return False
        elif self.websocket_connected:
            logger.info("Websocket already connected")
            return True
        return False

    async def subscribe_symbols(self, symbols: List[str]):
        if not self.websocket_connected:
            logger.warning("Websocket not connected. Cannot subscribe to symbols.")
            return False

        # Convert symbols to Fyers format
        from data.tickers import get_fyers_symbol
        fyers_symbols = [get_fyers_symbol(symbol) for symbol in symbols]

        logger.info(f"Subscribing to {len(fyers_symbols)} symbols (first 3: {fyers_symbols[:3]})")
        
        # Fyers websocket subscription format - use 'SymbolUpdate' for live data
        data_type = "SymbolUpdate"  # Correct data type for live market data
        
        try:
            self.data_ws.subscribe(symbols=fyers_symbols, data_type=data_type)
            self.subscribed_symbols.update(fyers_symbols)
            
            logger.info(f"Subscribed to {len(fyers_symbols)} symbols")
            self.ws_logger.info(f"SUBSCRIBED: {fyers_symbols}")
            
            # Give it a moment for data to start flowing
            await asyncio.sleep(2)
            
            return True
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
            return False

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        return self.price_data.get(symbol)

    async def get_market_microstructure(self, symbol: str) -> Dict:
        """Get market microstructure data (order book, volume profile)"""
        # This method will now rely on websocket data if available, or fall back to REST if needed
        # For simplicity, we'll just return the latest price from websocket for now
        latest_price = await self.get_latest_price(symbol)
        if latest_price:
            return {
                'last_price': latest_price,
                'timestamp': datetime.now()
            }
        # Fallback to REST API if websocket data is not available
        quotes = await self.get_live_quotes([symbol])
        if quotes and symbol in quotes:
            return {
                'last_price': quotes[symbol]['v']['lp'],
                'timestamp': datetime.fromtimestamp(quotes[symbol]['v']['tt'])
            }
        return {}

    async def detect_significant_moves(self, symbols: List[str]) -> List[Tuple[str, float]]:
        """Detect symbols with significant price moves (≥2%)"""
        try:
            # If market is not open, clear the cache
            if not self._is_market_open():
                self.previous_close_cache = {}

            # Get current prices from websocket data
            current_quotes = {}
            for symbol in symbols:
                latest_price = await self.get_latest_price(symbol)
                if latest_price:
                    current_quotes[symbol] = {'v': {'lp': latest_price}} # Mimic REST API response structure

            significant_moves = []
            
            for symbol in symbols:
                if symbol in current_quotes:
                    quote = current_quotes[symbol]
                    current_price = quote['v']['lp']  # Last price

                    previous_close = self.previous_close_cache.get(symbol)

                    if not previous_close:
                        # Fetch previous close using historical data for accurate calculation
                        historical_df = await self.get_historical_data(symbol, "D", 1) # Daily data for previous day
                        if not historical_df.empty:
                            previous_close = historical_df['close'].iloc[-1]
                            self.previous_close_cache[symbol] = previous_close

                    if previous_close and previous_close > 0:
                        move_percent = ((current_price - previous_close) / previous_close) * 100
                        
                        if abs(move_percent) >= Config.MIN_PRICE_MOVE_PERCENT:
                            significant_moves.append((symbol, move_percent))
                            
            return significant_moves
            
        except Exception as e:
            logger.error(f"Error detecting significant moves: {e}")
            return []

    def _is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            now = datetime.now()
            current_time = now.strftime('%H:%M')
            
            # Simple market hours check (9:15 AM to 3:30 PM IST)
            market_start = Config.MARKET_HOURS['start']
            market_end = Config.MARKET_HOURS['end']
            
            return market_start <= current_time <= market_end
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False

    async def calculate_price_features(self, symbol: str, lookback_periods: int = 5) -> np.ndarray:
        """Calculate price feature vector for cross-modal analysis"""
        try:
            df = await self.get_historical_data(symbol, timeframe="1", period=lookback_periods)
            if df.empty:
                return np.zeros(5) # Return a zero vector if no data

            # Calculate features
            price_change = (df['close'] - df['open']) / df['open']
            high_low_range = (df['high'] - df['low']) / df['low']
            volume_change = df['volume'].pct_change().fillna(0)
            
            # Create feature vector
            feature_vector = np.array([
                price_change.mean(),
                high_low_range.mean(),
                volume_change.mean(),
                df['close'].iloc[-1],
                df['volume'].iloc[-1]
            ])
            
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_vector

        except Exception as e:
            logger.error(f"Error calculating price features for {symbol}: {e}")
            return np.zeros(5) # Return a zero vector on error

    async def place_order(self, symbol: str, side: str, quantity: int, 
                         order_type: str = "MARKET") -> Dict:
        """Place trading order"""
        try:
            if not self.fyers:
                await self.initialize()
            
            # Convert symbol to Fyers format
            from data.tickers import get_fyers_symbol
            fyers_symbol = get_fyers_symbol(symbol)
            
            order_data = {
                "symbol": fyers_symbol,
                "qty": quantity,
                "type": 1 if side == "BUY" else -1,
                "side": 1 if side == "BUY" else -1,
                "productType": "INTRADAY",
                "limitPrice": 0,
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": "False",
                "stopLoss": 0,
                "takeProfit": 0
            }
            
            response = await self.fyers.place_order(order_data)
            logger.info(f"Order placed: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"code": -1, "message": str(e)}
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            if not self.fyers:
                await self.initialize()
            
            response = await self.fyers.positions()
            if response['code'] == 200:
                # Handle different response structures
                if 'overall' in response and 'netPositions' in response['overall']:
                    return response['overall']['netPositions']
                elif 'netPositions' in response:
                    return response['netPositions']
                elif 'data' in response:
                    return response['data']
                else:
                    logger.warning(f"Unexpected positions response structure: {response}")
                    return []
            return []
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_account_summary(self) -> Dict:
        """Get account summary"""
        try:
            if not self.fyers:
                await self.initialize()
            
            response = await self.fyers.funds()
            if response['code'] == 200:
                return response['fund_limits']
            return {}
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}

