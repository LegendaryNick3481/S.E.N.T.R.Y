"""
Realistic backtesting engine that simulates everything:
- Historical news with realistic timing
- Sentiment analysis delays
- Order execution with slippage
- Market impact and liquidity
- Commissions and fees
- Intraday price movements
"""
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

from backtesting.order_book import Order, OrderType, OrderSide
from backtesting.execution_simulator import ExecutionSimulator, BrokerConfig
from backtesting.portfolio import Portfolio

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest"""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0  # ₹1 lakh
    max_positions: int = 10
    position_size_pct: float = 0.10  # 10% per position
    
    # Signal thresholds
    min_discord_score: float = 0.3
    min_confidence: float = 0.6
    
    # Risk management
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    max_holding_days: int = 5  # Max 5 days
    
    # Realism parameters
    news_processing_delay_minutes: int = 5  # 5 min to process news
    sentiment_analysis_delay_minutes: int = 2  # 2 min for sentiment
    signal_generation_delay_minutes: int = 1  # 1 min to generate signal


@dataclass
class SimulatedNewsEvent:
    """Simulated news event with realistic timing"""
    symbol: str
    timestamp: datetime
    sentiment: float  # -1 to 1
    relevance: float  # 0 to 1
    title: str
    processed_at: datetime  # When sentiment was analyzed
    signal_generated_at: datetime  # When trading signal was generated


@dataclass
class BacktestSignal:
    """Trading signal from strategy"""
    symbol: str
    timestamp: datetime
    direction: OrderSide
    confidence: float
    discord_score: float
    price_at_signal: float
    news_sentiment: float


class RealisticBacktest:
    """Realistic backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = Portfolio(config.initial_capital, config.max_positions)
        self.executor = ExecutionSimulator(BrokerConfig())
        
        self.signals: List[BacktestSignal] = []
        self.news_events: List[SimulatedNewsEvent] = []
        
        self.current_time: datetime = None
        self.market_data: Dict[str, pd.DataFrame] = {}
        
    async def load_historical_data(self, symbols: List[str]):
        """
        Load historical OHLCV data for symbols
        
        In production, this would fetch from Fyers API
        For now, we'll generate realistic synthetic data
        """
        logger.info(f"Loading historical data for {len(symbols)} symbols")
        
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        
        for symbol in symbols:
            # Generate realistic price data
            df = self._generate_realistic_price_data(symbol, start, end)
            self.market_data[symbol] = df
            
        logger.info(f"Loaded data from {start} to {end}")
    
    def _generate_realistic_price_data(self, symbol: str, start: datetime, 
                                       end: datetime) -> pd.DataFrame:
        """
        Generate realistic OHLCV data with proper characteristics:
        - Trending behavior
        - Volatility clustering
        - Realistic volume patterns
        """
        # Generate date range (trading days only)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        
        # Starting price (random but realistic for Indian stocks)
        base_price = np.random.uniform(100, 2000)
        
        # Generate returns with realistic properties
        # Drift (slight upward bias)
        drift = 0.0005
        
        # Volatility (changes over time)
        base_vol = 0.02
        vol_persistence = 0.9
        volatility = np.zeros(n)
        volatility[0] = base_vol
        
        for i in range(1, n):
            # GARCH-like volatility clustering
            volatility[i] = base_vol + vol_persistence * (volatility[i-1] - base_vol) + \
                           np.random.normal(0, 0.002)
            volatility[i] = max(0.01, min(0.05, volatility[i]))  # Bound between 1-5%
        
        # Generate returns
        returns = np.random.normal(drift, volatility)
        
        # Add momentum/trending
        momentum = np.zeros(n)
        for i in range(1, n):
            momentum[i] = 0.3 * returns[i-1] + 0.7 * momentum[i-1]
        
        returns += momentum * 0.5
        
        # Generate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Intraday range (0.5% to 2%)
            intraday_range = np.random.uniform(0.005, 0.02) * close
            
            # Generate OHLC
            open_price = close * np.random.uniform(0.99, 1.01)
            high = max(open_price, close) + np.random.uniform(0, intraday_range)
            low = min(open_price, close) - np.random.uniform(0, intraday_range)
            
            # Volume (higher on volatile days)
            base_volume = np.random.uniform(50000, 200000)
            vol_factor = 1 + abs(returns[i]) * 10
            volume = int(base_volume * vol_factor)
            
            data.append({
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume,
                'volatility': volatility[i],
                'returns': returns[i]
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _generate_news_events(self, symbol: str, dates: pd.DatetimeIndex):
        """
        Generate realistic news events for a symbol
        
        News characteristics:
        - Random timing throughout day
        - Sentiment correlated with price movements
        - Realistic delays in processing
        """
        for date in dates:
            # Random number of news events per day (0-3)
            num_events = np.random.poisson(0.5)  # Average 0.5 events per day
            
            for _ in range(num_events):
                # Random time during trading hours (9:15 AM to 3:30 PM)
                hour = np.random.randint(9, 15)
                minute = np.random.randint(0, 60)
                
                if hour == 9 and minute < 15:
                    minute = 15
                elif hour == 15 and minute > 30:
                    minute = 30
                
                timestamp = date.replace(hour=hour, minute=minute)
                
                # Get price data for this day
                if date in self.market_data[symbol].index:
                    day_data = self.market_data[symbol].loc[date]
                    price_change = day_data['returns']
                    
                    # Sentiment somewhat correlated with price movement
                    # But with noise (mismatch opportunity)
                    base_sentiment = price_change * 10  # Scale up
                    noise = np.random.normal(0, 0.3)
                    sentiment = np.clip(base_sentiment + noise, -1, 1)
                    
                    # Relevance (random)
                    relevance = np.random.uniform(0.5, 1.0)
                    
                    # Processing delays
                    processed_at = timestamp + timedelta(
                        minutes=self.config.news_processing_delay_minutes + 
                               self.config.sentiment_analysis_delay_minutes
                    )
                    signal_generated_at = processed_at + timedelta(
                        minutes=self.config.signal_generation_delay_minutes
                    )
                    
                    event = SimulatedNewsEvent(
                        symbol=symbol,
                        timestamp=timestamp,
                        sentiment=sentiment,
                        relevance=relevance,
                        title=f"News event for {symbol}",
                        processed_at=processed_at,
                        signal_generated_at=signal_generated_at
                    )
                    
                    self.news_events.append(event)
    
    def _detect_mismatched_energy(self, symbol: str, timestamp: datetime, 
                                   news_sentiment: float) -> Optional[BacktestSignal]:
        """
        Detect mismatch between news and price
        
        This is the core strategy logic
        """
        # Get recent price action (lookback window)
        try:
            df = self.market_data[symbol]
            current_idx = df.index.get_indexer([timestamp], method='nearest')[0]
            
            if current_idx < 5:  # Need at least 5 periods
                return None
            
            # Price movement over last 5 periods
            recent_data = df.iloc[current_idx-5:current_idx+1]
            price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1)
            
            # Normalize to -1 to 1
            price_direction = np.clip(price_change * 10, -1, 1)
            
            # Calculate discord (misalignment)
            discord_score = abs(news_sentiment - price_direction) / 2
            
            # Confidence based on sentiment strength and volume
            volume_ratio = recent_data['volume'].iloc[-1] / recent_data['volume'].mean()
            confidence = (abs(news_sentiment) + min(volume_ratio, 2) / 2) / 2
            
            # Check thresholds
            if discord_score < self.config.min_discord_score:
                return None
            if confidence < self.config.min_confidence:
                return None
            
            # Generate signal
            # If sentiment positive but price down -> BUY (contrarian)
            # If sentiment negative but price up -> SELL (contrarian)
            if news_sentiment > 0 and price_direction < 0:
                direction = OrderSide.BUY
            elif news_sentiment < 0 and price_direction > 0:
                direction = OrderSide.SELL
            else:
                return None  # No mismatch
            
            signal = BacktestSignal(
                symbol=symbol,
                timestamp=timestamp,
                direction=direction,
                confidence=confidence,
                discord_score=discord_score,
                price_at_signal=recent_data['close'].iloc[-1],
                news_sentiment=news_sentiment
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error detecting mismatch for {symbol}: {e}")
            return None
    
    async def run(self, symbols: List[str]) -> Dict:
        """Run the backtest"""
        logger.info(f"Starting backtest for {len(symbols)} symbols")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Initial capital: ₹{self.config.initial_capital:,.2f}")
        
        # Load data
        await self.load_historical_data(symbols)
        
        # Generate news events for all symbols
        logger.info("Generating news events...")
        for symbol in symbols:
            dates = self.market_data[symbol].index
            self._generate_news_events(symbol, dates)
        
        logger.info(f"Generated {len(self.news_events)} news events")
        
        # Sort news events by signal generation time
        self.news_events.sort(key=lambda x: x.signal_generated_at)
        
        # Simulate trading
        logger.info("Simulating trading...")
        
        for event in self.news_events:
            self.current_time = event.signal_generated_at
            
            # Update market data and order books
            self._update_market_state(event.signal_generated_at)
            
            # Detect mismatch and generate signal
            signal = self._detect_mismatched_energy(
                event.symbol,
                event.signal_generated_at,
                event.sentiment
            )
            
            if signal:
                self.signals.append(signal)
                # Execute trade
                await self._execute_signal(signal)
            
            # Check exit conditions for existing positions
            await self._check_exits()
        
        # Close all positions at end
        await self._close_all_positions()
        
        # Generate results
        results = self._generate_results()
        
        logger.info("Backtest complete!")
        return results
    
    def _update_market_state(self, timestamp: datetime):
        """Update order books and portfolio values"""
        current_prices = {}
        
        for symbol, df in self.market_data.items():
            try:
                # Get price at this time
                idx = df.index.get_indexer([timestamp], method='ffill')[0]
                if idx >= 0:
                    row = df.iloc[idx]
                    current_prices[symbol] = row['close']
                    
                    # Update order book
                    self.executor.update_market_data(
                        symbol,
                        row['close'],
                        row['volatility'],
                        row['volume']
                    )
            except:
                pass
        
        # Update portfolio
        self.portfolio.update_prices(current_prices, timestamp.timestamp())
    
    async def _execute_signal(self, signal: BacktestSignal):
        """Execute a trading signal"""
        # Calculate position size
        portfolio_value = self.portfolio.get_total_value()
        position_value = portfolio_value * self.config.position_size_pct
        
        # Get current price
        df = self.market_data[signal.symbol]
        idx = df.index.get_indexer([signal.timestamp], method='nearest')[0]
        current_price = df.iloc[idx]['close']
        
        quantity = int(position_value / current_price)
        
        if quantity == 0:
            return
        
        # Create order
        order = Order(
            symbol=signal.symbol,
            side=signal.direction,
            order_type=OrderType.MARKET,
            quantity=quantity,
            timestamp=signal.timestamp.timestamp()
        )
        
        # Execute
        fill = self.executor.execute_order(order, signal.timestamp)
        
        if fill:
            success = self.portfolio.process_fill(fill)
            if success:
                logger.info(f"Signal executed: {signal.direction.value.upper()} {quantity} {signal.symbol} @ ₹{current_price:.2f}")
    
    async def _check_exits(self):
        """Check exit conditions for all positions"""
        if not self.current_time:
            return
        
        to_exit = []
        
        for symbol, position in self.portfolio.positions.items():
            # Get current price
            df = self.market_data[symbol]
            idx = df.index.get_indexer([self.current_time], method='nearest')[0]
            current_price = df.iloc[idx]['close']
            
            # Calculate P&L
            pnl_pct = (current_price - position.avg_entry_price) / position.avg_entry_price
            
            # Check stop loss
            if pnl_pct <= -self.config.stop_loss_pct:
                to_exit.append((symbol, "Stop Loss"))
                continue
            
            # Check take profit
            if pnl_pct >= self.config.take_profit_pct:
                to_exit.append((symbol, "Take Profit"))
                continue
            
            # Check max holding period
            entry_time = datetime.fromtimestamp(position.entry_time)
            holding_days = (self.current_time - entry_time).days
            if holding_days >= self.config.max_holding_days:
                to_exit.append((symbol, "Max Holding"))
                continue
        
        # Execute exits
        for symbol, reason in to_exit:
            await self._exit_position(symbol, reason)
    
    async def _exit_position(self, symbol: str, reason: str):
        """Exit a position"""
        position = self.portfolio.positions.get(symbol)
        if not position:
            return
        
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=position.quantity
        )
        
        fill = self.executor.execute_order(order, self.current_time)
        
        if fill:
            self.portfolio.process_fill(fill)
            logger.info(f"Position exited: {symbol} | Reason: {reason}")
    
    async def _close_all_positions(self):
        """Close all remaining positions at end of backtest"""
        logger.info("Closing all remaining positions...")
        
        symbols = list(self.portfolio.positions.keys())
        for symbol in symbols:
            await self._exit_position(symbol, "End of Backtest")
    
    def _generate_results(self) -> Dict:
        """Generate backtest results"""
        total_return = self.portfolio.get_return_pct()
        sharpe = self.portfolio.calculate_sharpe_ratio()
        max_dd = self.portfolio.calculate_max_drawdown()
        win_rate = self.portfolio.get_win_rate()
        avg_win, avg_loss = self.portfolio.get_avg_win_loss()
        
        results = {
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'initial_capital': self.config.initial_capital,
            'final_capital': self.portfolio.get_total_value(),
            'total_return_pct': total_return,
            'total_pnl': self.portfolio.get_total_pnl(),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': self.portfolio.num_trades,
            'num_closed_positions': len(self.portfolio.closed_positions),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_commissions': self.portfolio.total_commissions,
            'num_signals_generated': len(self.signals),
            'snapshots': self.portfolio.snapshots,
            'closed_positions': self.portfolio.closed_positions
        }
        
        return results
