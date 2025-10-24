"""
Portfolio management for backtesting
Tracks positions, cash, P&L, and performance metrics
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

from backtesting.order_book import OrderSide
from backtesting.execution_simulator import Fill

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a symbol"""
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commissions: float = 0.0
    entry_time: float = 0.0
    
    def update_price(self, price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = price
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity
    
    def get_market_value(self) -> float:
        """Get current market value of position"""
        return self.current_price * self.quantity
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized - commissions)"""
        return self.realized_pnl + self.unrealized_pnl - self.total_commissions


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio at a point in time"""
    timestamp: float
    cash: float
    positions_value: float
    total_value: float
    realized_pnl: float
    unrealized_pnl: float
    total_commissions: float
    num_positions: int
    num_trades: int


class Portfolio:
    """Portfolio manager for backtesting"""
    
    def __init__(self, initial_capital: float, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        self.total_commissions = 0.0
        self.total_realized_pnl = 0.0
        self.num_trades = 0
        
        # Performance tracking
        self.snapshots: List[PortfolioSnapshot] = []
        self.daily_returns: List[float] = []
        
    def process_fill(self, fill: Fill) -> bool:
        """
        Process a fill and update portfolio
        
        Returns:
            True if successful, False if insufficient funds/shares
        """
        symbol = fill.symbol
        
        if fill.side == OrderSide.BUY:
            return self._process_buy(fill)
        else:
            return self._process_sell(fill)
    
    def _process_buy(self, fill: Fill) -> bool:
        """Process a buy fill"""
        total_cost = (fill.price * fill.quantity) + fill.commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for {fill.symbol}: need ₹{total_cost:,.2f}, have ₹{self.cash:,.2f}")
            return False
        
        # Check position limits
        if fill.symbol not in self.positions and len(self.positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions}), cannot open new position in {fill.symbol}")
            return False
        
        # Deduct cash
        self.cash -= total_cost
        self.total_commissions += fill.commission
        self.num_trades += 1
        
        # Update or create position
        if fill.symbol in self.positions:
            pos = self.positions[fill.symbol]
            # Average in
            total_qty = pos.quantity + fill.quantity
            total_cost_basis = (pos.avg_entry_price * pos.quantity) + (fill.price * fill.quantity)
            pos.avg_entry_price = total_cost_basis / total_qty
            pos.quantity = total_qty
            pos.total_commissions += fill.commission
        else:
            # New position
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=fill.quantity,
                avg_entry_price=fill.price,
                current_price=fill.price,
                entry_time=fill.timestamp,
                total_commissions=fill.commission
            )
        
        logger.info(f"BUY {fill.quantity} {fill.symbol} @ ₹{fill.price:.2f} | Commission: ₹{fill.commission:.2f}")
        return True
    
    def _process_sell(self, fill: Fill) -> bool:
        """Process a sell fill"""
        # Check if we have the position
        if fill.symbol not in self.positions:
            logger.warning(f"Cannot sell {fill.symbol}: no position")
            return False
        
        pos = self.positions[fill.symbol]
        
        # Check if we have enough shares
        if fill.quantity > pos.quantity:
            logger.warning(f"Cannot sell {fill.quantity} {fill.symbol}: only have {pos.quantity}")
            return False
        
        # Calculate realized P&L
        proceeds = fill.price * fill.quantity
        cost_basis = pos.avg_entry_price * fill.quantity
        realized_pnl = proceeds - cost_basis - fill.commission
        
        # Update portfolio
        self.cash += proceeds - fill.commission
        self.total_commissions += fill.commission
        self.total_realized_pnl += realized_pnl
        self.num_trades += 1
        
        pos.quantity -= fill.quantity
        pos.realized_pnl += realized_pnl
        pos.total_commissions += fill.commission
        
        logger.info(f"SELL {fill.quantity} {fill.symbol} @ ₹{fill.price:.2f} | P&L: ₹{realized_pnl:,.2f}")
        
        # Close position if fully sold
        if pos.quantity == 0:
            self.closed_positions.append(pos)
            del self.positions[fill.symbol]
            logger.info(f"Position closed: {fill.symbol} | Total P&L: ₹{pos.get_total_pnl():,.2f}")
        
        return True
    
    def update_prices(self, prices: Dict[str, float], timestamp: float):
        """Update current prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
        
        # Take snapshot
        self._take_snapshot(timestamp)
    
    def _take_snapshot(self, timestamp: float):
        """Take a portfolio snapshot"""
        positions_value = sum(pos.get_market_value() for pos in self.positions.values())
        total_value = self.cash + positions_value
        
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_commissions=self.total_commissions,
            num_positions=len(self.positions),
            num_trades=self.num_trades
        )
        
        self.snapshots.append(snapshot)
        
        # Calculate daily return
        if len(self.snapshots) > 1:
            prev_value = self.snapshots[-2].total_value
            if prev_value > 0:
                daily_return = (total_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
    
    def get_total_value(self) -> float:
        """Get total portfolio value"""
        positions_value = sum(pos.get_market_value() for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_total_pnl(self) -> float:
        """Get total P&L"""
        return self.get_total_value() - self.initial_capital
    
    def get_return_pct(self) -> float:
        """Get total return percentage"""
        return (self.get_total_value() - self.initial_capital) / self.initial_capital * 100
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(self.daily_returns)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        # Annualize
        sharpe_annualized = sharpe * np.sqrt(252)
        
        return sharpe_annualized
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.snapshots) < 2:
            return 0.0
        
        values = [s.total_value for s in self.snapshots]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def get_win_rate(self) -> float:
        """Calculate win rate from closed positions"""
        if not self.closed_positions:
            return 0.0
        
        winning = sum(1 for pos in self.closed_positions if pos.get_total_pnl() > 0)
        return winning / len(self.closed_positions) * 100
    
    def get_avg_win_loss(self) -> tuple:
        """Get average win and loss amounts"""
        if not self.closed_positions:
            return (0.0, 0.0)
        
        wins = [pos.get_total_pnl() for pos in self.closed_positions if pos.get_total_pnl() > 0]
        losses = [pos.get_total_pnl() for pos in self.closed_positions if pos.get_total_pnl() < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        return (avg_win, avg_loss)
