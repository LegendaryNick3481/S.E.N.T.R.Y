"""
Realistic order book simulator for backtesting
Simulates bid-ask spread, market depth, and order matching
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None  # For limit orders
    timestamp: float = 0.0
    order_id: str = ""


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: float
    commission: float
    slippage: float


class OrderBook:
    """Simulates realistic order book with spread and depth"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: List[Tuple[float, int]] = []  # (price, quantity)
        self.asks: List[Tuple[float, int]] = []  # (price, quantity)
        
    def update_from_market_data(self, mid_price: float, volatility: float, volume: int):
        """
        Generate realistic order book based on market data
        
        Args:
            mid_price: Current mid price
            volatility: Recent volatility (used for spread calculation)
            volume: Recent volume (used for depth calculation)
        """
        # Calculate realistic spread based on volatility
        # Higher volatility = wider spread
        base_spread_bps = 5  # 5 basis points base spread
        volatility_spread_bps = volatility * 100  # Add volatility component
        total_spread_bps = base_spread_bps + volatility_spread_bps
        spread = mid_price * (total_spread_bps / 10000)
        
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Generate realistic depth
        # More volume = more liquidity
        # Increased depth to prevent order rejections
        base_depth = max(int(volume / 100), 500)
        
        # Generate 5 levels of bids and asks
        self.bids = []
        self.asks = []
        
        for level in range(5):
            # Bids decrease in price, increase in quantity
            bid_price = best_bid - (level * spread * 0.5)
            bid_qty = int(base_depth * (1 + level * 0.3))
            self.bids.append((bid_price, bid_qty))
            
            # Asks increase in price, increase in quantity
            ask_price = best_ask + (level * spread * 0.5)
            ask_qty = int(base_depth * (1 + level * 0.3))
            self.asks.append((ask_price, ask_qty))
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return self.bids[0][0] if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return self.asks[0][0] if self.asks else None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def match_market_order(self, side: OrderSide, quantity: int) -> Tuple[float, int]:
        """
        Match a market order against the book
        
        Returns:
            (average_fill_price, filled_quantity)
        """
        if side == OrderSide.BUY:
            # Buy market order matches against asks
            levels = self.asks
        else:
            # Sell market order matches against bids
            levels = self.bids
        
        total_filled = 0
        total_cost = 0.0
        remaining = quantity
        
        for price, qty in levels:
            if remaining <= 0:
                break
            
            fill_qty = min(remaining, qty)
            total_filled += fill_qty
            total_cost += price * fill_qty
            remaining -= fill_qty
        
        if total_filled == 0:
            return 0.0, 0
        
        avg_price = total_cost / total_filled
        return avg_price, total_filled
    
    def calculate_market_impact(self, side: OrderSide, quantity: int) -> float:
        """
        Calculate market impact for a large order
        
        Market impact increases with order size relative to liquidity
        """
        if side == OrderSide.BUY:
            levels = self.asks
        else:
            levels = self.bids
        
        # Calculate total available liquidity in top 5 levels
        total_liquidity = sum(qty for _, qty in levels)
        
        if total_liquidity == 0:
            return 0.10  # 10% impact if no liquidity
        
        # Impact as percentage of quantity to liquidity
        impact_ratio = quantity / total_liquidity
        
        # Non-linear impact: square root function
        # Small orders: minimal impact
        # Large orders: significant impact
        impact_pct = np.sqrt(impact_ratio) * 0.05  # Up to 5% for 100% of liquidity
        
        return min(impact_pct, 0.15)  # Cap at 15%
