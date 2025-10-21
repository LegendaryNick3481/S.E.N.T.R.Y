"""
Cross-modal analysis system for detecting news-price mismatches
The core "synesthesia detector" for mismatched energy
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from datetime import datetime, timedelta
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)

class CrossModalAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.correlation_history = {}
        self.discord_patterns = {}
        
    def analyze_cross_modal_correlation(self, news_embeddings: np.ndarray, 
                                      price_features: np.ndarray) -> Dict:
        """Analyze correlation between news embeddings and price features"""
        try:
            if news_embeddings.size == 0 or price_features.size == 0:
                return {
                    'correlation': 0.0,
                    'discord_score': 0.0,
                    'alignment': 'unknown',
                    'confidence': 0.0
                }
            
            # Ensure arrays are 2D for correlation calculation
            if news_embeddings.ndim == 1:
                news_embeddings = news_embeddings.reshape(1, -1)
            if price_features.ndim == 1:
                price_features = price_features.reshape(1, -1)
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(news_embeddings, price_features)[0][0]
            
            # Calculate correlation coefficient
            if len(news_embeddings) > 1 and len(price_features) > 1:
                correlation, p_value = pearsonr(news_embeddings.flatten(), price_features.flatten())
            else:
                correlation = cosine_sim
                p_value = 0.5
            
            # Determine alignment
            if correlation > 0.3:
                alignment = 'positive'
            elif correlation < -0.3:
                alignment = 'negative'
            else:
                alignment = 'neutral'
            
            # Calculate discord score (inverse of correlation for mismatched energy)
            discord_score = abs(correlation) if correlation < 0 else 0
            
            return {
                'correlation': correlation,
                'discord_score': discord_score,
                'alignment': alignment,
                'confidence': 1.0 - p_value,
                'cosine_similarity': cosine_sim,
                'p_value': p_value
            }
            
        except Exception as e:
            logger.error(f"Error in cross-modal correlation analysis: {e}")
            return {
                'correlation': 0.0,
                'discord_score': 0.0,
                'alignment': 'unknown',
                'confidence': 0.0
            }
    
    def detect_mismatched_energy(self, news_sentiment: float, price_direction: float,
                                volume_anomaly: float, volatility_spike: float) -> Dict:
        """Detect mismatched energy between news sentiment and price action"""
        try:
            # Calculate expected vs actual direction
            expected_direction = 1.0 if news_sentiment > Config.SENTIMENT_THRESHOLD else -1.0 if news_sentiment < -Config.SENTIMENT_THRESHOLD else 0.0
            actual_direction = 1.0 if price_direction > 0 else -1.0 if price_direction < 0 else 0.0
            
            # Calculate mismatch magnitude
            direction_mismatch = abs(expected_direction - actual_direction)
            
            # Calculate discord score components
            sentiment_price_discord = direction_mismatch * abs(news_sentiment)
            volume_discord = volume_anomaly * 0.5  # Weight volume anomaly
            volatility_discord = volatility_spike * 0.3  # Weight volatility spike
            
            # Combined discord score
            discord_score = (
                sentiment_price_discord * Config.DISCORD_WEIGHTS['sentiment_price_correlation'] +
                volume_discord * Config.DISCORD_WEIGHTS['volume_anomaly'] +
                volatility_discord * Config.DISCORD_WEIGHTS['volatility_spike']
            )
            
            # Determine mismatch type
            if direction_mismatch > 1.5:  # Complete opposite
                mismatch_type = 'complete_opposite'
            elif direction_mismatch > 0.5:  # Partial mismatch
                mismatch_type = 'partial_mismatch'
            else:
                mismatch_type = 'aligned'
            
            # Calculate confidence based on multiple factors
            confidence = min(1.0, (
                abs(news_sentiment) * 0.4 +  # News sentiment strength
                abs(price_direction) * 0.3 +  # Price movement strength
                volume_anomaly * 0.2 +  # Volume confirmation
                volatility_spike * 0.1  # Volatility confirmation
            ))
            
            return {
                'discord_score': discord_score,
                'mismatch_type': mismatch_type,
                'direction_mismatch': direction_mismatch,
                'expected_direction': expected_direction,
                'actual_direction': actual_direction,
                'confidence': confidence,
                'sentiment_price_discord': sentiment_price_discord,
                'volume_discord': volume_discord,
                'volatility_discord': volatility_discord,
                'is_mismatched': discord_score > Config.MIN_CORRELATION_THRESHOLD
            }
            
        except Exception as e:
            logger.error(f"Error detecting mismatched energy: {e}")
            return {
                'discord_score': 0.0,
                'mismatch_type': 'unknown',
                'direction_mismatch': 0.0,
                'confidence': 0.0,
                'is_mismatched': False
            }
    
    def calculate_volume_anomaly(self, current_volume: float, 
                               historical_volumes: List[float]) -> float:
        """Calculate volume anomaly score"""
        try:
            if not historical_volumes or current_volume <= 0:
                return 0.0
            
            # Calculate volume statistics
            mean_volume = np.mean(historical_volumes)
            std_volume = np.std(historical_volumes)
            
            if std_volume == 0:
                return 0.0
            
            # Calculate z-score
            z_score = (current_volume - mean_volume) / std_volume
            
            # Convert to anomaly score (0-1)
            anomaly_score = min(1.0, max(0.0, abs(z_score) / 3.0))  # Cap at 3 standard deviations
            
            return anomaly_score
            
        except Exception as e:
            logger.error(f"Error calculating volume anomaly: {e}")
            return 0.0
    
    def calculate_volatility_spike(self, current_volatility: float,
                                 historical_volatilities: List[float]) -> float:
        """Calculate volatility spike score"""
        try:
            if not historical_volatilities or current_volatility <= 0:
                return 0.0
            
            # Calculate volatility statistics
            mean_volatility = np.mean(historical_volatilities)
            std_volatility = np.std(historical_volatilities)
            
            if std_volatility == 0:
                return 0.0
            
            # Calculate z-score
            z_score = (current_volatility - mean_volatility) / std_volatility
            
            # Convert to spike score (0-1)
            spike_score = min(1.0, max(0.0, abs(z_score) / 2.0))  # Cap at 2 standard deviations
            
            return spike_score
            
        except Exception as e:
            logger.error(f"Error calculating volatility spike: {e}")
            return 0.0
    
    def analyze_market_microstructure(self, microstructure_data: Dict) -> Dict:
        """Analyze market microstructure for additional signals"""
        try:
            if not microstructure_data:
                return {
                    'bid_ask_spread': 0.0,
                    'order_imbalance': 0.0,
                    'liquidity_score': 0.0,
                    'market_impact': 0.0
                }
            
            # Calculate bid-ask spread
            bid_price = microstructure_data.get('bid_price', 0)
            ask_price = microstructure_data.get('ask_price', 0)
            last_price = microstructure_data.get('last_price', 0)
            
            if bid_price > 0 and ask_price > 0 and last_price > 0:
                spread = (ask_price - bid_price) / last_price
            else:
                spread = 0.0
            
            # Calculate order imbalance
            bid_quantity = microstructure_data.get('bid_quantity', 0)
            ask_quantity = microstructure_data.get('ask_quantity', 0)
            
            if bid_quantity > 0 or ask_quantity > 0:
                order_imbalance = (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity)
            else:
                order_imbalance = 0.0
            
            # Calculate liquidity score
            total_quantity = bid_quantity + ask_quantity
            liquidity_score = min(1.0, total_quantity / 10000)  # Normalize to reasonable scale
            
            # Calculate market impact (simplified)
            volume = microstructure_data.get('volume', 0)
            market_impact = min(1.0, volume / 100000)  # Normalize to reasonable scale
            
            return {
                'bid_ask_spread': spread,
                'order_imbalance': order_imbalance,
                'liquidity_score': liquidity_score,
                'market_impact': market_impact
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market microstructure: {e}")
            return {
                'bid_ask_spread': 0.0,
                'order_imbalance': 0.0,
                'liquidity_score': 0.0,
                'market_impact': 0.0
            }
    
    def calculate_discord_ranking(self, symbols_data: List[Dict]) -> List[Dict]:
        """Calculate discord ranking for multiple symbols"""
        try:
            ranked_symbols = []
            
            for symbol_data in symbols_data:
                symbol = symbol_data['symbol']
                discord_score = symbol_data.get('discord_score', 0.0)
                confidence = symbol_data.get('confidence', 0.0)
                volume_anomaly = symbol_data.get('volume_anomaly', 0.0)
                volatility_spike = symbol_data.get('volatility_spike', 0.0)
                
                # Calculate composite score
                composite_score = (
                    discord_score * 0.5 +
                    confidence * 0.3 +
                    volume_anomaly * 0.1 +
                    volatility_spike * 0.1
                )
                
                ranked_symbols.append({
                    'symbol': symbol,
                    'discord_score': discord_score,
                    'confidence': confidence,
                    'composite_score': composite_score,
                    'volume_anomaly': volume_anomaly,
                    'volatility_spike': volatility_spike,
                    'rank': 0  # Will be set after sorting
                })
            
            # Sort by composite score (descending)
            ranked_symbols.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Assign ranks
            for i, symbol_data in enumerate(ranked_symbols):
                symbol_data['rank'] = i + 1
            
            return ranked_symbols
            
        except Exception as e:
            logger.error(f"Error calculating discord ranking: {e}")
            return []
    
    def detect_pattern_anomalies(self, symbol: str, historical_data: List[Dict]) -> Dict:
        """Detect pattern anomalies in historical discord data"""
        try:
            if len(historical_data) < 10:  # Need sufficient history
                return {
                    'pattern_anomaly': False,
                    'anomaly_score': 0.0,
                    'pattern_type': 'insufficient_data'
                }
            
            # Extract discord scores over time
            discord_scores = [data.get('discord_score', 0.0) for data in historical_data]
            
            # Calculate rolling statistics
            window_size = min(5, len(discord_scores))
            rolling_mean = np.mean(discord_scores[-window_size:])
            rolling_std = np.std(discord_scores[-window_size:])
            
            # Current discord score
            current_discord = discord_scores[-1] if discord_scores else 0.0
            
            # Detect anomaly
            if rolling_std > 0:
                z_score = abs(current_discord - rolling_mean) / rolling_std
                anomaly_score = min(1.0, z_score / 3.0)  # Cap at 3 standard deviations
            else:
                anomaly_score = 0.0
            
            # Determine pattern type
            if anomaly_score > 0.7:
                pattern_type = 'extreme_anomaly'
            elif anomaly_score > 0.4:
                pattern_type = 'moderate_anomaly'
            else:
                pattern_type = 'normal'
            
            return {
                'pattern_anomaly': anomaly_score > 0.5,
                'anomaly_score': anomaly_score,
                'pattern_type': pattern_type,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
                'current_discord': current_discord
            }
            
        except Exception as e:
            logger.error(f"Error detecting pattern anomalies: {e}")
            return {
                'pattern_anomaly': False,
                'anomaly_score': 0.0,
                'pattern_type': 'error'
            }

