import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, Activity } from 'lucide-react';

interface MarketData {
  symbol: string;
  timestamp: string;
  mismatch_analysis?: {
    discord_score: number;
    is_mismatched: boolean;
    confidence: number;
  };
  sentiment_summary?: {
    weighted_sentiment: number;
    news_count: number;
    confidence: number;
  };
}

interface MarketOverviewProps {
  overview?: { symbols: Record<string, MarketData> };
  loading?: boolean;
}

export function MarketOverview({ overview, loading }: MarketOverviewProps) {
  const getTrendIcon = (discordScore: number) => {
    if (discordScore > 0.3) return TrendingUp;
    if (discordScore < -0.3) return TrendingDown;
    return Minus;
  };

  const getSignalClass = (discordScore: number) => {
    if (discordScore > 0.3) return 'signal-high';
    if (discordScore < -0.3) return 'signal-low';
    return 'border-l-4 border-gray-500 bg-gray-500/10';
  };

  const getSentimentClass = (sentiment: number) => {
    if (sentiment > 0.1) return 'text-success';
    if (sentiment < -0.1) return 'text-danger';
    return 'text-gray-400';
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
  };

  if (loading) {
    return (
      <div className="space-y-4">
        {[...Array(8)].map((_, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="glass-card p-4"
          >
            <div className="loading-shimmer h-16 rounded-xl" />
          </motion.div>
        ))}
      </div>
    );
  }

  const marketData = overview?.symbols ? Object.entries(overview.symbols) : [];

  return (
    <div className="space-y-3 max-h-96 overflow-y-auto scrollbar-hide">
      <AnimatePresence>
        {marketData.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12 text-gray-400"
          >
            <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No market data available</p>
            <p className="text-sm mt-2">Waiting for market analysis...</p>
          </motion.div>
        ) : (
          marketData.slice(0, 15).map(([symbol, data], index) => {
            const discordScore = data?.mismatch_analysis?.discord_score || 0;
            const sentiment = data?.sentiment_summary?.weighted_sentiment || 0;
            const TrendIcon = getTrendIcon(discordScore);
            
            return (
              <motion.div
                key={symbol}
                initial={{ opacity: 0, x: -50, scale: 0.9 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 50, scale: 0.9 }}
                transition={{ 
                  delay: index * 0.05,
                  duration: 0.4,
                  type: 'spring',
                  stiffness: 200
                }}
                whileHover={{ 
                  scale: 1.02,
                  transition: { duration: 0.2 }
                }}
                className={`glass-card p-4 ${getSignalClass(discordScore)}`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-white/10 rounded-lg">
                      <TrendIcon className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg">{symbol}</h3>
                      <p className="text-xs text-gray-400">
                        {data?.timestamp ? new Date(data.timestamp).toLocaleTimeString() : ''}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`font-bold text-lg ${getSentimentClass(sentiment)}`}>
                      {formatPercent(sentiment)}
                    </div>
                    <div className="text-xs text-gray-400">Sentiment</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-400 mb-1">Discord Score</div>
                    <div className="font-bold text-lg">
                      {discordScore.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 mb-1">News Count</div>
                    <div className="font-bold text-lg">
                      {data?.sentiment_summary?.news_count || 0}
                    </div>
                  </div>
                </div>

                {/* Discord score progress bar */}
                <div className="mt-3">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Discord Level</span>
                    <span>{Math.abs(discordScore * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-700/50 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(Math.abs(discordScore) * 100, 100)}%` }}
                      transition={{ delay: index * 0.05 + 0.3, duration: 0.6 }}
                      className={`h-2 rounded-full ${
                        discordScore > 0.3 ? 'bg-danger' :
                        discordScore < -0.3 ? 'bg-success' :
                        'bg-warning'
                      }`}
                    />
                  </div>
                </div>
              </motion.div>
            );
          })
        )}
      </AnimatePresence>
    </div>
  );
}
