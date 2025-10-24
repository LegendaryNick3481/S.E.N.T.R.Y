"""
Realistic trade execution simulator
Handles slippage, commissions, partial fills, and order timing
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import logging

from backtesting.order_book import Order, OrderBook, OrderType, OrderSide, Fill

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    """Broker-specific parameters"""
    commission_pct: float = 0.0003  # 0.03% (Fyers/Zerodha typical)
    min_commission: float = 0.0  # No minimum
    max_commission: float = 20.0  # ₹20 max per order
    
    # Slippage model parameters
    base_slippage_bps: float = 2.0  # 2 basis points base slippage
    volatility_multiplier: float = 1.5  # Slippage increases with volatility
    
    # Execution delays (realistic timing)
    order_processing_delay_ms: float = 100  # 100ms to process order
    market_order_fill_delay_ms: float = 200  # 200ms to fill market order
    
    # Market constraints
    max_order_size_pct: float = 0.05  # Can't be more than 5% of volume
    partial_fill_threshold: float = 0.7  # 70% fill rate for large orders


class ExecutionSimulator:
    """Simulates realistic order execution with slippage and fees"""
    
    def __init__(self, broker_config: BrokerConfig = None):
        self.config = broker_config or BrokerConfig()
        self.order_books: Dict[str, OrderBook] = {}
        self.order_counter = 0
        
    def update_market_data(self, symbol: str, price: float, volatility: float, volume: int):
        """Update order book with current market data"""
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
        
        self.order_books[symbol].update_from_market_data(price, volatility, volume)
    
    def execute_order(self, order: Order, current_time: datetime) -> Optional[Fill]:
        """
        Execute an order with realistic simulation
        
        Returns:
            Fill object if successful, None if rejected
        """
        # Get order book
        if order.symbol not in self.order_books:
            logger.warning(f"No order book for {order.symbol}")
            return None
        
        order_book = self.order_books[order.symbol]
        
        # Simulate order processing delay
        execution_time = current_time + timedelta(
            milliseconds=self.config.order_processing_delay_ms
        )
        
        # Check order size constraints
        if not self._validate_order_size(order, order_book):
            logger.warning(f"Order rejected: size too large for {order.symbol}")
            return None
        
        # Execute based on order type
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, order_book, execution_time)
        else:
            return self._execute_limit_order(order, order_book, execution_time)
    
    def _validate_order_size(self, order: Order, order_book: OrderBook) -> bool:
        """Check if order size is reasonable"""
        # Get available liquidity
        if order.side == OrderSide.BUY:
            levels = order_book.asks
        else:
            levels = order_book.bids
        
        total_liquidity = sum(qty for _, qty in levels)
        
        if total_liquidity == 0:
            return False
        
        # Reject if order is more than max % of available liquidity
        if order.quantity > total_liquidity * self.config.max_order_size_pct:
            return False
        
        return True
    
    def _execute_market_order(self, order: Order, order_book: OrderBook, 
                              execution_time: datetime) -> Optional[Fill]:
        """Execute market order with realistic slippage"""
        
        # Add market order fill delay
        fill_time = execution_time + timedelta(
            milliseconds=self.config.market_order_fill_delay_ms
        )
        
        # Match order against book
        avg_price, filled_qty = order_book.match_market_order(order.side, order.quantity)
        
        if filled_qty == 0:
            logger.warning(f"Market order for {order.symbol} could not be filled")
            return None
        
        # Calculate slippage
        mid_price = order_book.get_mid_price()
        if mid_price is None:
            return None
        
        # Base slippage + volatility component
        spread = order_book.get_spread() or (mid_price * 0.0001)
        slippage_pct = self.config.base_slippage_bps / 10000
        
        # Add market impact for large orders
        impact = order_book.calculate_market_impact(order.side, order.quantity)
        
        if order.side == OrderSide.BUY:
            # Buy: slippage increases price
            slippage = avg_price * (slippage_pct + impact)
            final_price = avg_price + slippage
        else:
            # Sell: slippage decreases price
            slippage = avg_price * (slippage_pct + impact)
            final_price = avg_price - slippage
        
        # Calculate commission
        commission = self._calculate_commission(final_price, filled_qty)
        
        # Create fill
        self.order_counter += 1
        fill = Fill(
            order_id=f"ORDER_{self.order_counter:06d}",
            symbol=order.symbol,
            side=order.side,
            quantity=filled_qty,
            price=final_price,
            timestamp=fill_time.timestamp(),
            commission=commission,
            slippage=slippage
        )
        
        return fill
    
    def _execute_limit_order(self, order: Order, order_book: OrderBook,
                            execution_time: datetime) -> Optional[Fill]:
        """Execute limit order (only if price is favorable)"""
        
        if order.price is None:
            return None
        
        # Check if limit price would be filled
        if order.side == OrderSide.BUY:
            # Buy limit: only fill if ask <= limit price
            best_ask = order_book.get_best_ask()
            if best_ask is None or best_ask > order.price:
                return None
            fill_price = min(order.price, best_ask)
        else:
            # Sell limit: only fill if bid >= limit price
            best_bid = order_book.get_best_bid()
            if best_bid is None or best_bid < order.price:
                return None
            fill_price = max(order.price, best_bid)
        
        # Limit orders have less slippage (you control the price)
        slippage = 0.0
        
        # But might get partial fills for large orders
        filled_qty = order.quantity
        if order.quantity > self.config.partial_fill_threshold:
            # Simulate partial fill
            fill_rate = np.random.uniform(0.7, 1.0)
            filled_qty = int(order.quantity * fill_rate)
        
        # Calculate commission
        commission = self._calculate_commission(fill_price, filled_qty)
        
        # Create fill
        self.order_counter += 1
        fill = Fill(
            order_id=f"ORDER_{self.order_counter:06d}",
            symbol=order.symbol,
            side=order.side,
            quantity=filled_qty,
            price=fill_price,
            timestamp=execution_time.timestamp(),
            commission=commission,
            slippage=slippage
        )
        
        return fill
    
    def _calculate_commission(self, price: float, quantity: int) -> float:
        """Calculate trading commission"""
        trade_value = price * quantity
        commission = trade_value * self.config.commission_pct
        
        # Apply min and max
        commission = max(self.config.min_commission, commission)
        commission = min(self.config.max_commission, commission)
        
        return commission
