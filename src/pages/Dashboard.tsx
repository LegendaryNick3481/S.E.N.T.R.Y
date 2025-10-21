import React from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { 
  Wallet, 
  TrendingUp, 
  Target, 
  BarChart3,
  Activity,
  Brain
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { TradingSignals } from '../components/TradingSignals';
import { PortfolioSummary } from '../components/PortfolioSummary';
import { PerformanceChart } from '../components/PerformanceChart';
import { NewsFeed } from '../components/NewsFeed';
import { MarketOverview } from '../components/MarketOverview';

// API functions
const fetchOverview = async () => {
  const response = await fetch('/api/overview');
  if (!response.ok) throw new Error('Failed to fetch overview');
  return response.json();
};

const fetchSignals = async () => {
  const response = await fetch('/api/signals');
  if (!response.ok) throw new Error('Failed to fetch signals');
  return response.json();
};

const fetchPortfolio = async () => {
  const response = await fetch('/api/portfolio');
  if (!response.ok) throw new Error('Failed to fetch portfolio');
  return response.json();
};

export function Dashboard() {
  const { data: overview, isLoading: overviewLoading } = useQuery({ queryKey: ['overview'], queryFn: fetchOverview });
  const { data: signals, isLoading: signalsLoading } = useQuery({ queryKey: ['signals'], queryFn: fetchSignals });
  const { data: portfolio, isLoading: portfolioLoading } = useQuery({ queryKey: ['portfolio'], queryFn: fetchPortfolio });

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
    return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
  };

  const getTotalPnl = () => {
    if (!portfolio?.portfolio?.positions) return 0;
    return Object.values(portfolio.portfolio.positions).reduce((sum: number, pos: any) => {
      return sum + (pos.pnl || 0);
    }, 0);
  };

  const getActivePositions = () => {
    return Object.keys(portfolio?.portfolio?.positions || {}).length;
  };

  const getPortfolioChange = () => {
    const totalPnl = getTotalPnl();
    return formatPercent(totalPnl / (portfolio?.portfolio?.portfolio_value || 1));
  };

  const getPositionsPnl = () => {
    const totalPnl = getTotalPnl();
    return formatCurrency(totalPnl);
  };

  return (
    <div className="p-6 space-y-8">
      {/* Key Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        <MetricCard
          title="Portfolio Value"
          value={formatCurrency(portfolio?.portfolio?.portfolio_value)}
          change={getPortfolioChange()}
          icon={Wallet}
          loading={portfolioLoading}
          delay={0.1}
        />
        <MetricCard
          title="Active Positions"
          value={getActivePositions()}
          change={getPositionsPnl()}
          icon={Activity}
          loading={portfolioLoading}
          delay={0.2}
        />
        <MetricCard
          title="Avg Discord Score"
          value={overview?.market_summary?.avg_discord_score?.toFixed(3) || '-'}
          change={`${overview?.market_summary?.mismatched_symbols || 0} mismatched`}
          icon={Brain}
          loading={overviewLoading}
          delay={0.3}
        />
        <MetricCard
          title="Win Rate"
          value={formatPercent(overview?.win_rate)}
          change={`${overview?.total_trades || 0} trades`}
          icon={Target}
          loading={overviewLoading}
          delay={0.4}
        />
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Trading Signals */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="lg:col-span-2"
        >
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gradient">Trading Signals</h2>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-400">Live</span>
                <div className="w-2 h-2 bg-success rounded-full animate-pulse glow-effect" />
              </div>
            </div>
            <TradingSignals 
              signals={signals?.signals || []} 
              loading={signalsLoading} 
            />
          </div>
        </motion.div>

        {/* Portfolio & Performance */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="space-y-6"
        >
          {/* Portfolio Summary */}
          <div className="glass-card p-6">
            <h3 className="text-xl font-bold mb-6 text-gradient">Portfolio Summary</h3>
            <PortfolioSummary 
              portfolio={portfolio?.portfolio} 
              loading={portfolioLoading} 
            />
          </div>

          {/* Performance Chart */}
          <div className="glass-card p-6">
            <h3 className="text-xl font-bold mb-6 text-gradient">Performance</h3>
            <PerformanceChart 
              data={overview?.portfolio_history} 
              loading={overviewLoading} 
            />
          </div>
        </motion.div>
      </div>

      {/* News & Market Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* News Feed */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.5 }}
        >
          <div className="glass-card p-6">
            <h3 className="text-xl font-bold mb-6 text-gradient">Latest News</h3>
            <NewsFeed 
              news={overview?.news_data} 
              loading={overviewLoading} 
            />
          </div>
        </motion.div>

        {/* Market Overview */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.5 }}
        >
          <div className="glass-card p-6">
            <h3 className="text-xl font-bold mb-6 text-gradient">Market Overview</h3>
            <MarketOverview 
              overview={overview} 
              loading={overviewLoading} 
            />
          </div>
        </motion.div>
      </div>
    </div>
  );
}
