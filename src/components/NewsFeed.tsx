import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ExternalLink, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface NewsItem {
  headline: string;
  source: string;
  timestamp: string;
  sentiment: number;
  symbol?: string;
}

interface NewsFeedProps {
  news?: Record<string, NewsItem[]>;
  loading?: boolean;
}

export function NewsFeed({ news, loading }: NewsFeedProps) {
  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 0.1) return TrendingUp;
    if (sentiment < -0.1) return TrendingDown;
    return Minus;
  };

  const getSentimentClass = (sentiment: number) => {
    if (sentiment > 0.1) return 'text-success';
    if (sentiment < -0.1) return 'text-danger';
    return 'text-gray-400';
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
  };

  if (loading) {
    return (
      <div className="space-y-4">
        {[...Array(5)].map((_, i) => (
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

  // Flatten news data
  const allNews: NewsItem[] = [];
  if (news) {
    Object.values(news).forEach(symbolNews => {
      allNews.push(...symbolNews);
    });
  }

  // Sort by timestamp (newest first)
  allNews.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

  return (
    <div className="space-y-4 max-h-96 overflow-y-auto scrollbar-hide">
      <AnimatePresence>
        {allNews.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12 text-gray-400"
          >
            <ExternalLink className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No news available</p>
            <p className="text-sm mt-2">Waiting for market updates...</p>
          </motion.div>
        ) : (
          allNews.slice(0, 10).map((item, index) => {
            const SentimentIcon = getSentimentIcon(item.sentiment);
            return (
              <motion.div
                key={`${item.headline}-${index}`}
                initial={{ opacity: 0, x: -50, scale: 0.9 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 50, scale: 0.9 }}
                transition={{ 
                  delay: index * 0.1,
                  duration: 0.4,
                  type: 'spring',
                  stiffness: 200
                }}
                whileHover={{ 
                  scale: 1.02,
                  transition: { duration: 0.2 }
                }}
                className="glass-card p-4 hover:bg-white/5 transition-all duration-300"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    {item.symbol && (
                      <div className="text-xs text-primary-400 font-medium mb-1">
                        {item.symbol}
                      </div>
                    )}
                    <h4 className="text-sm font-medium text-white mb-2 line-clamp-2">
                      {item.headline}
                    </h4>
                    <div className="flex items-center space-x-3 text-xs text-gray-400">
                      <span>{item.source}</span>
                      <span>•</span>
                      <span>{formatTime(item.timestamp)}</span>
                    </div>
                  </div>
                  <div className="ml-4 flex flex-col items-center">
                    <SentimentIcon className={`w-5 h-5 ${getSentimentClass(item.sentiment)}`} />
                    <span className={`text-xs font-medium ${getSentimentClass(item.sentiment)}`}>
                      {formatPercent(item.sentiment)}
                    </span>
                  </div>
                </div>

                {/* Sentiment bar */}
                <div className="flex items-center space-x-2">
                  <div className="flex-1 h-1 bg-gray-700/50 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.abs(item.sentiment) * 100}%` }}
                      transition={{ delay: index * 0.1 + 0.5, duration: 0.8 }}
                      className={`h-full ${
                        item.sentiment > 0 ? 'bg-success' : 
                        item.sentiment < 0 ? 'bg-danger' : 
                        'bg-gray-400'
                      }`}
                    />
                  </div>
                  <span className={`text-xs font-medium ${getSentimentClass(item.sentiment)}`}>
                    {Math.abs(item.sentiment * 100).toFixed(0)}%
                  </span>
                </div>
              </motion.div>
            );
          })
        )}
      </AnimatePresence>
    </div>
  );
}
