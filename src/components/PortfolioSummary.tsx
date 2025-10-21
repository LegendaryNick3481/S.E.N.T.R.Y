import React from 'react';
import { motion } from 'framer-motion';
import { Wallet, TrendingUp, TrendingDown, Clock } from 'lucide-react';

interface Position {
  pnl: number;
  // Add other relevant properties from the position object
}

interface PortfolioData {
  portfolio_value?: number;
  available_cash?: number;
  exposure?: number;
  timestamp?: string;
  positions?: Record<string, Position>;
}

interface PortfolioSummaryProps {
  portfolio?: PortfolioData;
  loading?: boolean;
}

export function PortfolioSummary({ portfolio, loading }: PortfolioSummaryProps) {
  const formatCurrency = (amount?: number) => {
    if (typeof amount !== 'number') return '-';
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatPercent = (value?: number) => {
    if (typeof value !== 'number') return '-';
    return `${(value * 100).toFixed(1)}%`;
  };

  const getTotalPnl = () => {
    if (!portfolio?.positions) return 0;
    return Object.values(portfolio.positions).reduce((sum: number, pos: Position) => {
      return sum + (pos.pnl || 0);
    }, 0);
  };

  const portfolioItems = [
    {
      label: 'Total Value',
      value: formatCurrency(portfolio?.portfolio_value),
      icon: Wallet,
      color: 'text-primary-400',
    },
    {
      label: 'Available Cash',
      value: formatCurrency(portfolio?.available_cash),
      icon: Wallet,
      color: 'text-success',
    },
    {
      label: 'P&L',
      value: formatCurrency(getTotalPnl()),
      icon: getTotalPnl() >= 0 ? TrendingUp : TrendingDown,
      color: getTotalPnl() >= 0 ? 'text-success' : 'text-danger',
    },
    {
      label: 'Exposure',
      value: formatPercent(portfolio?.exposure),
      icon: TrendingUp,
      color: 'text-warning',
    },
    {
      label: 'Positions',
      value: Object.keys(portfolio?.positions || {}).length,
      icon: Wallet,
      color: 'text-blue-400',
    },
    {
      label: 'Last Updated',
      value: portfolio?.timestamp ? new Date(portfolio.timestamp).toLocaleTimeString() : '-',
      icon: Clock,
      color: 'text-gray-400',
    },
  ];

  if (loading) {
    return (
      <div className="space-y-4">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: i * 0.1 }}
            className="flex justify-between items-center"
          >
            <div className="loading-shimmer h-4 w-24 rounded" />
            <div className="loading-shimmer h-4 w-16 rounded" />
          </motion.div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {portfolioItems.map((item, index) => (
        <motion.div
          key={item.label}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1, duration: 0.4 }}
          className="flex items-center justify-between p-3 glass-card rounded-xl hover:bg-white/5 transition-all duration-300"
        >
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-white/10 rounded-lg">
              <item.icon className={`w-4 h-4 ${item.color}`} />
            </div>
            <span className="text-gray-300 font-medium">{item.label}</span>
          </div>
          <span className={`font-bold ${item.color}`}>
            {item.value}
          </span>
        </motion.div>
      ))}
    </div>
  );
}
