"""
Backtesting engine for Mismatched Energy trading system
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import logging
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
import asyncio

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self):
        self.initial_capital = Config.INITIAL_CAPITAL
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.news_data = {}
        self.price_data = {}
        
    async def load_historical_data(self, symbols: List[str], 
                                 start_date: str, end_date: str) -> Dict:
        """Load historical price and news data"""
        try:
            # This would integrate with Fyers API to get historical data
            # For now, we'll create mock data structure
            
            price_data = {}
            news_data = {}
            
            for symbol in symbols:
                # Mock price data (in real implementation, fetch from Fyers)
                price_data[symbol] = self._generate_mock_price_data(symbol, start_date, end_date)
                
                # Mock news data (in real implementation, fetch from news sources)
                news_data[symbol] = self._generate_mock_news_data(symbol, start_date, end_date)
            
            self.price_data = price_data
            self.news_data = news_data
            
            logger.info(f"Loaded historical data for {len(symbols)} symbols")
            return {'price_data': price_data, 'news_data': news_data}
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return {}
    
    def _generate_mock_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock price data for backtesting"""
        try:
            # Create date range - use 5-minute intervals for faster backtesting
            dates = pd.date_range(start=start_date, end=end_date, freq='5min')
            
            # Generate mock OHLC data with some randomness
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            
            # Base price
            base_price = 100 + hash(symbol) % 1000
            
            # Generate price series with very conservative trend and volatility
            returns = np.random.normal(0.00001, 0.001, len(dates))  # Minimal drift, 0.1% volatility
            
            # Add minimal momentum and mean reversion
            for i in range(1, len(returns)):
                returns[i] += 0.001 * returns[i-1]  # Minimal momentum
                returns[i] -= 0.0005 * np.sum(returns[:i]) / i  # Minimal mean reversion
            
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate OHLC from prices with realistic volatility
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Generate OHLC around the price with minimal volatility
                volatility = 0.0005  # 0.05% intraday volatility
                high = price * (1 + np.random.uniform(0, volatility))
                low = price * (1 - np.random.uniform(0, volatility))
                open_price = prices[i-1] if i > 0 else price
                close = price
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock price data: {e}")
            return pd.DataFrame()
    
    def _generate_mock_news_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Generate mock news data for backtesting"""
        try:
            news_items = []
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Generate news items at random intervals
            current_date = start_dt
            while current_date < end_dt:
                # Random interval between news items (1-7 days)
                interval = np.random.randint(1, 8)
                current_date += timedelta(days=interval)
                
                if current_date > end_dt:
                    break
                
                # Generate mock news with sentiment
                sentiment = np.random.choice(['positive', 'negative', 'neutral'], 
                                          p=[0.3, 0.3, 0.4])
                
                if sentiment == 'positive':
                    title = f"{symbol} shows strong growth prospects"
                    sentiment_score = np.random.uniform(0.1, 0.8)
                elif sentiment == 'negative':
                    title = f"{symbol} faces challenges ahead"
                    sentiment_score = np.random.uniform(-0.8, -0.1)
                else:
                    title = f"{symbol} maintains steady performance"
                    sentiment_score = np.random.uniform(-0.1, 0.1)
                
                news_items.append({
                    'timestamp': current_date,
                    'title': title,
                    'description': f"Market update for {symbol}",
                    'sentiment_score': sentiment_score,
                    'source': 'mock_news'
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error generating mock news data: {e}")
            return []
    
    async def run_backtest(self, symbols: List[str], start_date: str, end_date: str,
                          strategy_params: Dict = None) -> Dict:
        """Run backtest for the mismatched energy strategy"""
        try:
            logger.info(f"Starting backtest for {len(symbols)} symbols from {start_date} to {end_date}")
            
            # Load historical data
            await self.load_historical_data(symbols, start_date, end_date)
            
            # Initialize backtest state
            self.current_capital = self.initial_capital
            self.positions = {symbol: 0 for symbol in symbols}
            self.trades = []
            self.portfolio_history = []
            
            # Get all unique timestamps
            all_timestamps = set()
            for symbol in symbols:
                if symbol in self.price_data:
                    all_timestamps.update(self.price_data[symbol].index)
            
            timestamps = sorted(list(all_timestamps))
            
            # Run backtest for each timestamp with progress bar
            total_timestamps = len(timestamps)
            logger.info(f"Processing {total_timestamps} timestamps...")
            
            # Create progress bar with minimal updates
            progress_bar = tqdm(total=total_timestamps, desc="Backtesting", unit="timestamps", 
                              disable=False, leave=True, ncols=80)
            
            for i, timestamp in enumerate(timestamps):
                await self._process_timestamp(timestamp, symbols, strategy_params)
                
                # Record portfolio value
                portfolio_value = self._calculate_portfolio_value(timestamp, symbols)
                self.portfolio_history.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value,
                    'cash': self.current_capital,
                    'positions': self.positions.copy()
                })
                
                # Update progress bar
                progress_bar.update(1)
                
                # Log progress every 10000 timestamps with strategy info
                if i % 10000 == 0 and i > 0:
                    logger.info(f"Processed {i+1}/{total_timestamps} timestamps ({((i+1)/total_timestamps)*100:.1f}%)")
                    logger.info(f"Current trades: {len(self.trades)}, Portfolio: {self.current_capital:.2f}")
            
            progress_bar.close()
            
            # Calculate results
            results = self._calculate_backtest_results()
            
            logger.info(f"Backtest completed. Final portfolio value: {results['final_value']:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    async def _process_timestamp(self, timestamp: datetime, symbols: List[str], 
                                strategy_params: Dict = None):
        """Process a single timestamp in the backtest"""
        try:
            # Get current prices for all symbols
            current_prices = {}
            for symbol in symbols:
                if symbol in self.price_data:
                    symbol_data = self.price_data[symbol]
                    if timestamp in symbol_data.index:
                        current_prices[symbol] = symbol_data.loc[timestamp, 'close']
            
            # Get news data for this timestamp
            news_data = self._get_news_for_timestamp(timestamp, symbols)
            
            # Simulate the mismatched energy strategy
            # This would integrate with the actual strategy components
            discord_scores = self._calculate_discord_scores(timestamp, symbols, news_data, current_prices)
            
            # Generate trading signals
            signals = self._generate_signals(discord_scores, current_prices)
            
            # Execute trades
            await self._execute_trades(signals, current_prices, timestamp)
            
        except Exception as e:
            logger.error(f"Error processing timestamp {timestamp}: {e}")
    
    def _get_news_for_timestamp(self, timestamp: datetime, symbols: List[str]) -> Dict:
        """Get news data for a specific timestamp"""
        news_data = {}
        
        for symbol in symbols:
            if symbol in self.news_data:
                # Get news within 1 hour of timestamp
                time_window = timedelta(hours=1)
                relevant_news = [
                    news for news in self.news_data[symbol]
                    if abs((news['timestamp'] - timestamp).total_seconds()) <= time_window.total_seconds()
                ]
                news_data[symbol] = relevant_news
        
        return news_data
    
    def _calculate_discord_scores(self, timestamp: datetime, symbols: List[str], 
                                news_data: Dict, current_prices: Dict) -> Dict:
        """Calculate discord scores for symbols"""
        discord_scores = {}
        
        for symbol in symbols:
            if symbol not in current_prices:
                continue
            
            # Get price change
            price_change = self._get_price_change(symbol, timestamp)
            
            # Get news sentiment
            news_sentiment = self._get_news_sentiment(news_data.get(symbol, []))
            
            # Calculate discord (simplified)
            expected_direction = 1 if news_sentiment > 0.05 else -1 if news_sentiment < -0.05 else 0
            actual_direction = 1 if price_change > 0.003 else -1 if price_change < -0.003 else 0
            
            discord = abs(expected_direction - actual_direction) * abs(news_sentiment)
            discord_scores[symbol] = {
                'discord_score': discord,
                'price_change': price_change,
                'news_sentiment': news_sentiment,
                'confidence': 0.8  # Mock confidence
            }
        
        return discord_scores
    
    def _get_price_change(self, symbol: str, timestamp: datetime) -> float:
        """Get price change for a symbol at timestamp"""
        try:
            if symbol not in self.price_data:
                return 0.0
            
            symbol_data = self.price_data[symbol]
            if timestamp not in symbol_data.index:
                return 0.0
            
            # Get price change from previous close
            current_price = symbol_data.loc[timestamp, 'close']
            
            # Find previous close
            prev_timestamp = symbol_data.index[symbol_data.index < timestamp]
            if len(prev_timestamp) > 0:
                prev_price = symbol_data.loc[prev_timestamp[-1], 'close']
                return (current_price - prev_price) / prev_price
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting price change: {e}")
            return 0.0
    
    def _get_news_sentiment(self, news_items: List[Dict]) -> float:
        """Calculate average news sentiment"""
        if not news_items:
            return 0.0
        
        sentiments = [item.get('sentiment_score', 0.0) for item in news_items]
        return np.mean(sentiments)
    
    def _generate_signals(self, discord_scores: Dict, current_prices: Dict) -> List[Dict]:
        """Generate trading signals based on discord scores"""
        signals = []
        
        # Sort by discord score
        sorted_symbols = sorted(discord_scores.items(), 
                              key=lambda x: x[1]['discord_score'], reverse=True)
        
        # Debug logging
        if len(sorted_symbols) > 0:
            max_discord = max(data['discord_score'] for data in discord_scores.values())
            logger.debug(f"Max discord score: {max_discord:.3f}, Symbols with discord > 0.1: {sum(1 for data in discord_scores.values() if data['discord_score'] > 0.1)}")
        
        for symbol, data in sorted_symbols:
            if data['discord_score'] > 0.05:  # Lower threshold for more trading opportunities
                # Determine signal direction based on price change
                price_change = data['price_change']
                if price_change > 0.003:  # Lower threshold for price changes
                    action = 'SELL'
                elif price_change < -0.003:  # Lower threshold for price changes
                    action = 'BUY'
                else:
                    continue
                
                # Calculate position size (slightly larger but still safe)
                position_value = self.current_capital * 0.02 * data['discord_score']  # 2% max
                quantity = max(1, int(position_value / max(1e-6, current_prices[symbol])))
                
                signals.append({
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': current_prices[symbol],
                    'discord_score': data['discord_score']
                })
        
        return signals
    
    async def _execute_trades(self, signals: List[Dict], current_prices: Dict, timestamp: datetime):
        """Execute trading signals"""
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            quantity = signal['quantity']
            price = signal['price']
            
            if action == 'BUY':
                cost = quantity * price
                if cost <= self.current_capital:
                    self.current_capital -= cost
                    self.positions[symbol] += quantity
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': price,
                        'value': cost
                    })
            
            elif action == 'SELL':
                if self.positions[symbol] >= quantity:
                    proceeds = quantity * price
                    self.current_capital += proceeds
                    self.positions[symbol] -= quantity
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': price,
                        'value': proceeds
                    })
    
    def _calculate_portfolio_value(self, timestamp: datetime, symbols: List[str]) -> float:
        """Calculate total portfolio value at timestamp"""
        total_value = self.current_capital
        
        for symbol in symbols:
            if symbol in self.positions and self.positions[symbol] > 0:
                if symbol in self.price_data and timestamp in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[timestamp, 'close']
                    total_value += self.positions[symbol] * price
        
        # Add bounds checking to prevent unrealistic values
        max_reasonable_value = self.initial_capital * 10  # Max 10x return
        if total_value > max_reasonable_value:
            logger.warning(f"Portfolio value {total_value:.2f} exceeds reasonable bounds, capping at {max_reasonable_value:.2f}")
            total_value = max_reasonable_value
        
        return total_value
    
    def _calculate_backtest_results(self) -> Dict:
        """Calculate backtest performance metrics"""
        try:
            if not self.portfolio_history:
                return {}
            
            # Extract portfolio values
            portfolio_values = [p['portfolio_value'] for p in self.portfolio_history]
            timestamps = [p['timestamp'] for p in self.portfolio_history]
            
            # Guard: need at least two points to compute returns
            if len(portfolio_values) < 2:
                initial_value = portfolio_values[0]
                final_value = portfolio_values[-1]
                return {
                    'initial_value': initial_value,
                    'final_value': final_value,
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'num_trades': len(self.trades),
                    'win_rate': 0.0,
                    'portfolio_history': self.portfolio_history,
                    'trades': self.trades
                }
            
            # Calculate per-step returns
            returns = np.diff(portfolio_values) / np.clip(portfolio_values[:-1], a_min=1e-9, a_max=None)
            
            # Basic metrics
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # Annualized return based on time delta (more robust than step count)
            try:
                total_days = max(1, (timestamps[-1] - timestamps[0]).days)
                annualized_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0.0
            except Exception:
                annualized_return = 0.0
            
            # Risk metrics (guard against empty/zero-variance arrays)
            ret_std = float(np.std(returns))
            volatility = ret_std * np.sqrt(252) if ret_std > 0 else 0.0
            sharpe_ratio = (float(np.mean(returns)) / ret_std * np.sqrt(252)) if ret_std > 0 else 0.0
            
            # Maximum drawdown (guard for empty arrays)
            if returns.size == 0:
                max_drawdown = 0.0
            else:
                cumulative_returns = np.cumprod(1 + returns)
                if cumulative_returns.size == 0:
                    max_drawdown = 0.0
                else:
                    running_max = np.maximum.accumulate(cumulative_returns)
                    # Avoid division by zero
                    running_max = np.clip(running_max, a_min=1e-9, a_max=None)
                    drawdowns = (cumulative_returns - running_max) / running_max
                    max_drawdown = float(np.min(drawdowns)) if drawdowns.size > 0 else 0.0
            
            # Trade analysis
            num_trades = len(self.trades)
            winning_trades = sum(1 for trade in self.trades if trade['action'] == 'SELL')
            win_rate = winning_trades / num_trades if num_trades > 0 else 0
            
            return {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'portfolio_history': self.portfolio_history,
                'trades': self.trades
            }
            
        except Exception as e:
            logger.error(f"Error calculating backtest results: {e}")
            return {}
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot backtest results"""
        try:
            if not results or 'portfolio_history' not in results:
                logger.error("No portfolio history to plot")
                return
            
            portfolio_history = results['portfolio_history']
            timestamps = [p['timestamp'] for p in portfolio_history]
            values = [p['portfolio_value'] for p in portfolio_history]
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Portfolio value over time
            ax1.plot(timestamps, values, label='Portfolio Value', linewidth=2)
            ax1.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_ylabel('Portfolio Value (INR)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Returns distribution
            returns = np.diff(values) / values[:-1]
            ax2.hist(returns, bins=50, alpha=0.7, label='Daily Returns')
            ax2.axvline(x=0, color='r', linestyle='--', label='Zero Return')
            ax2.set_title('Returns Distribution')
            ax2.set_xlabel('Daily Returns')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Results plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
    
    def generate_report(self, results: Dict) -> str:
        """Generate text report of backtest results"""
        try:
            if not results:
                return "No results to report"
            
            report = f"""
Mismatched Energy Strategy - Backtest Report
==========================================

Performance Summary:
- Initial Capital: INR {results.get('initial_value', 0):,.2f}
- Final Value: INR {results.get('final_value', 0):,.2f}
- Total Return: {results.get('total_return', 0)*100:.2f}%
- Annualized Return: {results.get('annualized_return', 0)*100:.2f}%

Risk Metrics:
- Volatility: {results.get('volatility', 0)*100:.2f}%
- Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
- Maximum Drawdown: {results.get('max_drawdown', 0)*100:.2f}%

Trading Activity:
- Total Trades: {results.get('num_trades', 0)}
- Win Rate: {results.get('win_rate', 0)*100:.2f}%

Strategy Performance:
- The strategy identified {results.get('num_trades', 0)} mismatched energy opportunities
- Average return per trade: {results.get('total_return', 0)/max(1, results.get('num_trades', 1))*100:.2f}%
- Risk-adjusted performance (Sharpe): {results.get('sharpe_ratio', 0):.2f}
"""
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "Error generating report"

