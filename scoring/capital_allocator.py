"""
Capital allocation and scoring system for mismatched energy trading
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class CapitalAllocator:
    def __init__(self):
        self.current_positions = {}
        self.portfolio_value = Config.INITIAL_CAPITAL
        self.risk_budget = Config.MAX_TOTAL_EXPOSURE
        self.position_sizes = {}
        self.performance_history = []
        
    def calculate_discord_score(self, symbol_data: Dict) -> float:
        """Calculate comprehensive discord score for a symbol"""
        try:
            # Extract components
            discord_score = symbol_data.get('discord_score', 0.0)
            confidence = symbol_data.get('confidence', 0.0)
            volume_anomaly = symbol_data.get('volume_anomaly', 0.0)
            volatility_spike = symbol_data.get('volatility_spike', 0.0)
            sentiment_mismatch = symbol_data.get('sentiment_price_mismatch', False)
            
            # Calculate base discord score
            base_score = discord_score * confidence
            
            # Add volume and volatility components
            volume_component = volume_anomaly * 0.3
            volatility_component = volatility_spike * 0.2
            
            # Sentiment mismatch bonus
            sentiment_bonus = 0.2 if sentiment_mismatch else 0.0
            
            # Calculate final discord score
            final_score = base_score + volume_component + volatility_component + sentiment_bonus
            
            # Normalize to 0-1 range
            final_score = min(1.0, max(0.0, final_score))
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating discord score: {e}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, discord_score: float, 
                              current_price: float, volatility: float) -> Dict:
        """Calculate optimal position size for a symbol"""
        try:
            # Base position size from discord score
            base_size_ratio = discord_score * Config.MAX_POSITION_SIZE
            
            # Adjust for volatility (higher volatility = smaller position)
            volatility_adjustment = 1.0 / (1.0 + volatility)
            adjusted_size_ratio = base_size_ratio * volatility_adjustment
            
            # Calculate position value
            position_value = self.portfolio_value * adjusted_size_ratio
            
            # Calculate number of shares (assuming integer shares)
            num_shares = int(position_value / current_price)
            actual_position_value = num_shares * current_price
            
            # Calculate position ratio
            position_ratio = actual_position_value / self.portfolio_value
            
            return {
                'symbol': symbol,
                'num_shares': num_shares,
                'position_value': actual_position_value,
                'position_ratio': position_ratio,
                'discord_score': discord_score,
                'volatility_adjustment': volatility_adjustment
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'symbol': symbol,
                'num_shares': 0,
                'position_value': 0.0,
                'position_ratio': 0.0,
                'discord_score': discord_score,
                'volatility_adjustment': 1.0
            }
    
    def optimize_portfolio_allocation(self, candidate_symbols: List[Dict]) -> List[Dict]:
        """Optimize portfolio allocation across multiple symbols"""
        try:
            if not candidate_symbols:
                return []
            
            # Filter symbols with minimum discord score
            min_discord_threshold = 0.3
            filtered_symbols = [
                symbol for symbol in candidate_symbols 
                if symbol.get('discord_score', 0.0) >= min_discord_threshold
            ]
            
            if not filtered_symbols:
                return []
            
            # Sort by discord score (descending)
            filtered_symbols.sort(key=lambda x: x['discord_score'], reverse=True)
            
            # Calculate allocations
            allocations = []
            total_exposure = 0.0
            
            for symbol_data in filtered_symbols:
                symbol = symbol_data['symbol']
                discord_score = symbol_data['discord_score']
                current_price = symbol_data.get('current_price', 0.0)
                volatility = symbol_data.get('volatility', 0.1)
                
                if current_price <= 0:
                    continue
                
                # Calculate position size
                position_info = self.calculate_position_size(
                    symbol, discord_score, current_price, volatility
                )
                
                # Check if we can add this position without exceeding limits
                new_exposure = total_exposure + position_info['position_ratio']
                
                if new_exposure <= self.risk_budget:
                    allocations.append({
                        **position_info,
                        'allocation_rank': len(allocations) + 1,
                        'cumulative_exposure': new_exposure
                    })
                    total_exposure = new_exposure
                else:
                    # Try with reduced position size
                    max_additional_exposure = self.risk_budget - total_exposure
                    if max_additional_exposure > 0.01:  # At least 1%
                        reduced_ratio = max_additional_exposure
                        reduced_shares = int((self.portfolio_value * reduced_ratio) / current_price)
                        reduced_value = reduced_shares * current_price
                        
                        if reduced_shares > 0:
                            allocations.append({
                                'symbol': symbol,
                                'num_shares': reduced_shares,
                                'position_value': reduced_value,
                                'position_ratio': reduced_ratio,
                                'discord_score': discord_score,
                                'volatility_adjustment': position_info['volatility_adjustment'],
                                'allocation_rank': len(allocations) + 1,
                                'cumulative_exposure': self.risk_budget
                            })
                            total_exposure = self.risk_budget
                            break
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio allocation: {e}")
            return []
    
    def calculate_risk_metrics(self, allocations: List[Dict]) -> Dict:
        """Calculate risk metrics for the portfolio"""
        try:
            if not allocations:
                return {
                    'total_exposure': 0.0,
                    'concentration_risk': 0.0,
                    'diversification_score': 0.0,
                    'max_position_ratio': 0.0,
                    'risk_score': 0.0
                }
            
            # Calculate total exposure
            total_exposure = sum(alloc['position_ratio'] for alloc in allocations)
            
            # Calculate concentration risk (Herfindahl index)
            position_ratios = [alloc['position_ratio'] for alloc in allocations]
            concentration_risk = sum(ratio ** 2 for ratio in position_ratios)
            
            # Calculate diversification score
            num_positions = len(allocations)
            max_diversification = 1.0 / num_positions if num_positions > 0 else 0.0
            diversification_score = 1.0 - concentration_risk / max_diversification if max_diversification > 0 else 0.0
            
            # Calculate max position ratio
            max_position_ratio = max(position_ratios) if position_ratios else 0.0
            
            # Calculate overall risk score
            exposure_risk = min(1.0, total_exposure / self.risk_budget)
            concentration_penalty = concentration_risk * 0.5
            risk_score = exposure_risk + concentration_penalty
            
            return {
                'total_exposure': total_exposure,
                'concentration_risk': concentration_risk,
                'diversification_score': diversification_score,
                'max_position_ratio': max_position_ratio,
                'risk_score': min(1.0, risk_score),
                'num_positions': num_positions
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'total_exposure': 0.0,
                'concentration_risk': 0.0,
                'diversification_score': 0.0,
                'max_position_ratio': 0.0,
                'risk_score': 0.0
            }
    
    def generate_trading_signals(self, allocations: List[Dict]) -> List[Dict]:
        """Generate trading signals from allocations"""
        try:
            signals = []
            
            for allocation in allocations:
                symbol = allocation['symbol']
                num_shares = allocation['num_shares']
                current_position = self.current_positions.get(symbol, 0)
                
                # Determine signal type
                if num_shares > current_position:
                    # Buy signal
                    signal_type = 'BUY'
                    quantity = num_shares - current_position
                elif num_shares < current_position:
                    # Sell signal
                    signal_type = 'SELL'
                    quantity = current_position - num_shares
                else:
                    # No change
                    continue
                
                if quantity > 0:
                    signal = {
                        'symbol': symbol,
                        'action': signal_type,
                        'quantity': quantity,
                        'discord_score': allocation['discord_score'],
                        'position_ratio': allocation['position_ratio'],
                        'timestamp': datetime.now(),
                        'signal_strength': allocation['discord_score']
                    }
                    signals.append(signal)
            
            # Sort signals by discord score (highest first)
            signals.sort(key=lambda x: x['discord_score'], reverse=True)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return []
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value"""
        self.portfolio_value = new_value
    
    def update_positions(self, symbol: str, quantity: int):
        """Update current positions"""
        self.current_positions[symbol] = quantity
    
    def calculate_portfolio_performance(self) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            if not self.performance_history:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
            
            # Calculate returns
            returns = [p['return'] for p in self.performance_history if 'return' in p]
            
            if not returns:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
            
            # Calculate metrics
            total_return = sum(returns)
            avg_return = np.mean(returns)
            return_std = np.std(returns)
            
            # Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = avg_return / return_std if return_std > 0 else 0.0
            
            # Maximum drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # Win rate
            winning_trades = sum(1 for r in returns if r > 0)
            win_rate = winning_trades / len(returns) if returns else 0.0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': len(returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
    
    def should_rebalance(self, current_allocations: List[Dict]) -> bool:
        """Determine if portfolio should be rebalanced"""
        try:
            # Check if any position has drifted significantly
            for allocation in current_allocations:
                symbol = allocation['symbol']
                target_ratio = allocation['position_ratio']
                current_ratio = self.current_positions.get(symbol, 0) / self.portfolio_value
                
                # Rebalance if position has drifted by more than 20%
                if abs(target_ratio - current_ratio) > 0.2:
                    return True
            
            # Check if total exposure has changed significantly
            current_exposure = sum(
                self.current_positions.get(alloc['symbol'], 0) / self.portfolio_value 
                for alloc in current_allocations
            )
            
            if abs(current_exposure - self.risk_budget) > 0.1:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rebalance condition: {e}")
            return False

