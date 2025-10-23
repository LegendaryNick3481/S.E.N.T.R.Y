"""
NLP processing system for news sentiment analysis
Uses sentence-transformers and VADER sentiment
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import logging
from datetime import datetime
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.sentence_model = None
        self.vader_analyzer = None
        self.news_embeddings_cache = {}
        self.sentiment_cache = {}
        
    async def initialize(self):
        """Initialize NLP models"""
        try:
            # Load sentence transformer model to CPU first
            self.sentence_model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL, device='cpu')
            
            # If CUDA is available, move the model to GPU
            if torch.cuda.is_available():
                self.sentence_model.to('cuda')
            
            # Initialize VADER sentiment analyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            logger.info("NLP models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            return False
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using VADER"""
        try:
            if not text or not text.strip():
                return {
                    'compound': 0.0,
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,
                    'label': 'neutral'
                }
            
            # Get VADER sentiment scores
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine label
            if scores['compound'] >= 0.05:
                label = 'positive'
            elif scores['compound'] <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'label': label
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'label': 'neutral'
            }
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding for text"""
        try:
            if not text or not text.strip():
                return np.zeros(384)  # Default embedding size for MiniLM
            
            # Check cache first
            cache_key = hash(text)
            if cache_key in self.news_embeddings_cache:
                return self.news_embeddings_cache[cache_key]
            
            # Generate embedding
            embedding = self.sentence_model.encode(text)
            
            # Cache the result
            self.news_embeddings_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return np.zeros(384)
    
    def analyze_news_batch(self, news_items: List[Dict]) -> List[Dict]:
        """Analyze sentiment for a batch of news items"""
        analyzed_news = []
        
        try:
            for news_item in news_items:
                # Combine title and description
                full_text = f"{news_item.get('title', '')} {news_item.get('description', '')}"
                
                # Analyze sentiment
                sentiment = self.analyze_sentiment(full_text)
                
                # Get embedding
                embedding = self.get_text_embedding(full_text)
                
                # Enhanced news item
                enhanced_item = {
                    **news_item,
                    'sentiment': sentiment,
                    'embedding': embedding,
                    'sentiment_score': sentiment['compound'],
                    'processed_at': datetime.now()
                }
                
                analyzed_news.append(enhanced_item)
                
        except Exception as e:
            logger.error(f"Error analyzing news batch: {e}")
            
        return analyzed_news
    
    def calculate_news_sentiment_summary(self, analyzed_news: List[Dict]) -> Dict:
        """Calculate overall sentiment summary for news"""
        try:
            if not analyzed_news:
                return {
                    'overall_sentiment': 0.0,
                    'sentiment_confidence': 0.0,
                    'positive_ratio': 0.0,
                    'negative_ratio': 0.0,
                    'neutral_ratio': 1.0,
                    'news_count': 0,
                    'weighted_sentiment': 0.0
                }
            
            # Extract sentiment scores
            sentiment_scores = [item['sentiment_score'] for item in analyzed_news]
            labels = [item['sentiment']['label'] for item in analyzed_news]
            
            # Calculate metrics
            overall_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            sentiment_confidence = 1.0 - sentiment_std  # Higher std = lower confidence
            
            # Calculate ratios
            total_count = len(analyzed_news)
            positive_count = labels.count('positive')
            negative_count = labels.count('negative')
            neutral_count = labels.count('neutral')
            
            positive_ratio = positive_count / total_count
            negative_ratio = negative_count / total_count
            neutral_ratio = neutral_count / total_count
            
            # Weighted sentiment (recent news gets higher weight)
            weights = np.exp(np.linspace(-1, 0, total_count))  # Exponential decay
            weighted_sentiment = np.average(sentiment_scores, weights=weights)
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_confidence': max(0.0, min(1.0, sentiment_confidence)),
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'neutral_ratio': neutral_ratio,
                'news_count': total_count,
                'weighted_sentiment': weighted_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error calculating news sentiment summary: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_confidence': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 1.0,
                'news_count': 0,
                'weighted_sentiment': 0.0
            }
    
    def detect_sentiment_anomalies(self, news_items: List[Dict], 
                                 price_direction: float) -> Dict:
        """Detect sentiment anomalies that could indicate mismatched energy"""
        try:
            if not news_items:
                return {
                    'anomaly_detected': False,
                    'anomaly_score': 0.0,
                    'sentiment_price_mismatch': False,
                    'confidence': 0.0
                }
            
            # Calculate news sentiment
            sentiment_summary = self.calculate_news_sentiment_summary(news_items)
            news_sentiment = sentiment_summary['weighted_sentiment']
            
            # Determine expected price direction from sentiment
            if news_sentiment > Config.SENTIMENT_THRESHOLD:
                expected_direction = 1.0  # Positive sentiment -> price up
            elif news_sentiment < -Config.SENTIMENT_THRESHOLD:
                expected_direction = -1.0  # Negative sentiment -> price down
            else:
                expected_direction = 0.0  # Neutral sentiment
            
            # Calculate mismatch
            actual_direction = 1.0 if price_direction > 0 else -1.0 if price_direction < 0 else 0.0
            mismatch = abs(expected_direction - actual_direction)
            
            # Detect anomaly
            anomaly_detected = mismatch > 1.0  # Complete opposite direction
            anomaly_score = mismatch * sentiment_summary['sentiment_confidence']
            
            return {
                'anomaly_detected': anomaly_detected,
                'anomaly_score': anomaly_score,
                'sentiment_price_mismatch': mismatch > 0.5,
                'confidence': sentiment_summary['sentiment_confidence'],
                'news_sentiment': news_sentiment,
                'expected_direction': expected_direction,
                'actual_direction': actual_direction,
                'mismatch_magnitude': mismatch
            }
            
        except Exception as e:
            logger.error(f"Error detecting sentiment anomalies: {e}")
            return {
                'anomaly_detected': False,
                'anomaly_score': 0.0,
                'sentiment_price_mismatch': False,
                'confidence': 0.0
            }
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from news text"""
        try:
            # Simple key phrase extraction (can be enhanced with more sophisticated methods)
            # Remove common words and extract meaningful phrases
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            phrases = []
            
            # Extract 2-3 word phrases
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5 and not any(word in stop_words for word in phrase.split()):
                    phrases.append(phrase)
            
            # Extract 3-word phrases
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(phrase) > 8 and not any(word in stop_words for word in phrase.split()):
                    phrases.append(phrase)
            
            # Return unique phrases, sorted by length
            unique_phrases = list(set(phrases))
            return sorted(unique_phrases, key=len, reverse=True)[:10]  # Top 10 phrases
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def calculate_sentiment_momentum(self, news_timeline: List[Dict]) -> Dict:
        """Calculate sentiment momentum over time"""
        try:
            if len(news_timeline) < 2:
                return {
                    'momentum': 0.0,
                    'trend': 'stable',
                    'acceleration': 0.0
                }
            
            # Sort by timestamp
            sorted_news = sorted(news_timeline, key=lambda x: x.get('published', datetime.now()))
            
            # Extract sentiment scores over time
            sentiments = [item['sentiment_score'] for item in sorted_news]
            
            # Calculate momentum (rate of change)
            momentum = np.gradient(sentiments)
            avg_momentum = np.mean(momentum)
            
            # Calculate acceleration (second derivative)
            acceleration = np.gradient(momentum)
            avg_acceleration = np.mean(acceleration)
            
            # Determine trend
            if avg_momentum > 0.1:
                trend = 'increasing'
            elif avg_momentum < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return {
                'momentum': avg_momentum,
                'trend': trend,
                'acceleration': avg_acceleration,
                'volatility': np.std(sentiments)
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment momentum: {e}")
            return {
                'momentum': 0.0,
                'trend': 'stable',
                'acceleration': 0.0
            }



# extract_key_phrases(self, text: str) -> List[str] --> needs to be better
# calculate_sentiment_momentum(self, news_timeline: List[Dict]) -> Dict --> where do we use this ?
