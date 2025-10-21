import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';

interface Signal {
  symbol: string;
  discord_score: number;
  confidence: number;
  sentiment: number;
  current_price: number;
  timestamp: string;
}

interface TradingSignalsProps {
  signals: Signal[];
  loading?: boolean;
}

export function TradingSignals({ signals, loading }: TradingSignalsProps) {
  const getSignalIcon = (discordScore: number) => {
    if (discordScore > 0.3) return AlertTriangle;
    if (discordScore > 0.1) return TrendingUp;
    return CheckCircle;
  };

  const getSignalClass = (discordScore: number) => {
    if (discordScore > 0.3) return 'signal-high';
    if (discordScore > 0.1) return 'signal-medium';
    return 'signal-low';
  };

  const getSentimentClass = (sentiment: number) => {
    if (sentiment > 0.1) return 'text-success';
    if (sentiment < -0.1) return 'text-danger';
    return 'text-gray-400';
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
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
            <div className="loading-shimmer h-20 rounded-xl" />
          </motion.div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4 max-h-96 overflow-y-auto scrollbar-hide">
      <AnimatePresence>
        {signals.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12 text-gray-400"
          >
            <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No trading signals available</p>
            <p className="text-sm mt-2">Waiting for market opportunities...</p>
          </motion.div>
        ) : (
          signals.slice(0, 10).map((signal, index) => {
            const SignalIcon = getSignalIcon(signal.discord_score);
            return (
              <motion.div
                key={`${signal.symbol}-${index}`}
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
                className={`glass-card p-4 ${getSignalClass(signal.discord_score)}`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-white/10 rounded-lg">
                      <SignalIcon className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg">{signal.symbol}</h3>
                      <p className="text-xs text-gray-400">{signal.timestamp}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-lg">
                      {formatCurrency(signal.current_price)}
                    </div>
                    <div className={`text-sm font-medium ${getSentimentClass(signal.sentiment)}`}>
                      {formatPercent(signal.sentiment)}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-xs text-gray-400 mb-1">Discord Score</div>
                    <div className="font-bold text-lg">
                      {signal.discord_score.toFixed(3)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-400 mb-1">Confidence</div>
                    <div className="font-bold text-lg">
                      {(signal.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-400 mb-1">Sentiment</div>
                    <div className={`font-bold text-lg ${getSentimentClass(signal.sentiment)}`}>
                      {formatPercent(signal.sentiment)}
                    </div>
                  </div>
                </div>

                {/* Progress bar for discord score */}
                <div className="mt-3">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Signal Strength</span>
                    <span>{Math.abs(signal.discord_score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-700/50 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(Math.abs(signal.discord_score) * 100, 100)}%` }}
                      transition={{ delay: index * 0.1 + 0.5, duration: 0.8 }}
                      className={`h-2 rounded-full ${
                        signal.discord_score > 0.3 ? 'bg-danger' :
                        signal.discord_score > 0.1 ? 'bg-warning' :
                        'bg-success'
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
