"""
Fyers API client for real-time market data and trading
"""
import asyncio
import json
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from fyers_apiv3 import fyersModel
from config import Config
import logging

logger = logging.getLogger(__name__)

class FyersClient:
    def __init__(self):
        self.fyers = None
        self.websocket = None
        self.is_connected = False
        self.subscribed_symbols = set()
        self.price_data = {}
        self.volume_data = {}
        
    async def initialize(self):
        """Initialize Fyers API connection"""
        try:
            self.fyers = fyersModel.FyersModel(
                client_id=Config.FYERS_APP_ID,
                token=Config.FYERS_ACCESS_TOKEN,
                log_path="logs/"
            )
            
            # Test connection
            profile = self.fyers.get_profile()
            if profile['code'] == 200:
                logger.info("Fyers API connection established")
                return True
            else:
                logger.error(f"Failed to connect to Fyers API: {profile}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Fyers client: {e}")
            return False
    
    async def get_historical_data(self, symbol: str, timeframe: str = "1", 
                                period: int = 1) -> pd.DataFrame:
        """Get historical OHLC data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period)
            
            data = {
                "symbol": symbol,
                "resolution": timeframe,
                "date_format": "1",
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": end_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data)
            
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
            
            response = self.fyers.quotes(data)
            
            if response['code'] == 200:
                return response['d']
            else:
                logger.error(f"Failed to get live quotes: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting live quotes: {e}")
            return {}
    
    async def detect_significant_moves(self, symbols: List[str]) -> List[Tuple[str, float]]:
        """Detect symbols with significant price moves (≥2%)"""
        try:
            # Get current and previous close prices
            current_quotes = await self.get_live_quotes(symbols)
            significant_moves = []
            
            for symbol in symbols:
                if symbol in current_quotes:
                    quote = current_quotes[symbol]
                    current_price = quote['v']['lp']  # Last price
                    previous_close = quote['v']['pc']  # Previous close
                    
                    if previous_close > 0:
                        move_percent = ((current_price - previous_close) / previous_close) * 100
                        
                        if abs(move_percent) >= Config.MIN_PRICE_MOVE_PERCENT:
                            significant_moves.append((symbol, move_percent))
                            
            return significant_moves
            
        except Exception as e:
            logger.error(f"Error detecting significant moves: {e}")
            return []
    
    async def get_market_microstructure(self, symbol: str) -> Dict:
        """Get market microstructure data (order book, volume profile)"""
        try:
            quote = await self.get_live_quotes([symbol])
            if symbol in quote:
                data = quote[symbol]['v']
                return {
                    'last_price': data['lp'],
                    'volume': data['v'],
                    'bid_price': data['bp'],
                    'ask_price': data['ap'],
                    'bid_quantity': data['bq'],
                    'ask_quantity': data['aq'],
                    'open_interest': data.get('oi', 0),
                    'timestamp': datetime.now()
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error getting microstructure data: {e}")
            return {}
    
    async def calculate_price_features(self, symbol: str, lookback_periods: int = 5) -> np.ndarray:
        """Calculate price feature vector for cross-modal analysis"""
        try:
            # Get recent price data
            df = await self.get_historical_data(symbol, "1", 1)  # 1 day, 1 minute data
            
            if df.empty:
                return np.array([])
            
            # Calculate features
            features = []
            
            # Price momentum
            returns = df['close'].pct_change().dropna()
            features.extend([
                returns.mean(),  # Average return
                returns.std(),   # Volatility
                returns.skew(),  # Skewness
                returns.kurtosis()  # Kurtosis
            ])
            
            # Volume features
            volume_returns = df['volume'].pct_change().dropna()
            features.extend([
                volume_returns.mean(),
                volume_returns.std(),
                df['volume'].mean(),
                df['volume'].std()
            ])
            
            # Technical indicators
            high_low_ratio = (df['high'] / df['low']).mean()
            close_open_ratio = (df['close'] / df['open']).mean()
            
            features.extend([high_low_ratio, close_open_ratio])
            
            # Price position within recent range
            recent_high = df['high'].max()
            recent_low = df['low'].min()
            current_price = df['close'].iloc[-1]
            price_position = (current_price - recent_low) / (recent_high - recent_low)
            features.append(price_position)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error calculating price features: {e}")
            return np.array([])
    
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
            
            response = self.fyers.place_order(order_data)
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
            
            response = self.fyers.positions()
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
            
            response = self.fyers.funds()
            if response['code'] == 200:
                return response['fund_limits']
            return {}
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}

