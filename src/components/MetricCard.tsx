import React from 'react';
import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: string;
  icon: LucideIcon;
  loading?: boolean;
  delay?: number;
}

export function MetricCard({ 
  title, 
  value, 
  change, 
  icon: Icon, 
  loading = false,
  delay = 0 
}: MetricCardProps) {
  return (
    <motion.div
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ 
        delay, 
        duration: 0.5, 
        type: 'spring',
        stiffness: 200,
        damping: 20
      }}
      whileHover={{ 
        scale: 1.02,
        y: -5,
        transition: { duration: 0.2 }
      }}
      className="metric-card glass-card-hover"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="p-3 bg-primary-500/20 rounded-xl">
          <Icon className="w-6 h-6 text-primary-400" />
        </div>
        {loading && (
          <div className="w-6 h-6 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
        )}
      </div>
      
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-gray-400">{title}</h3>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay + 0.3 }}
          className="text-2xl font-bold text-white"
        >
          {loading ? (
            <div className="h-8 bg-gray-700/50 rounded loading-shimmer" />
          ) : (
            value
          )}
        </motion.div>
        
        {change && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: delay + 0.5 }}
            className={`text-sm font-medium ${
              change.startsWith('+') ? 'text-success' : 
              change.startsWith('-') ? 'text-danger' : 
              'text-gray-400'
            }`}
          >
            {change}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
