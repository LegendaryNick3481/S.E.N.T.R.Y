import React from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Play, Bell, User } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

export function Header() {
  const queryClient = useQueryClient();

  const handleRefresh = () => {
    queryClient.invalidateQueries();
    toast.success('Data refreshed!');
  };

  const handleRunCycle = async () => {
    try {
      const response = await fetch('/api/run-cycle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (response.ok) {
        toast.success('Trading cycle started!');
        setTimeout(() => {
          queryClient.invalidateQueries();
        }, 2000);
      } else {
        toast.error('Failed to start trading cycle');
      }
    } catch (error) {
      toast.error('Error starting trading cycle');
    }
  };

  return (
    <motion.header
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="p-6 border-b border-border"
    >
      <div className="flex items-center justify-between">
        <div>
          <motion.h1
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="text-3xl font-bold text-gradient"
          >
            Trading Dashboard
          </motion.h1>
          <motion.p
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="text-gray-400 mt-1"
          >
            Real-time market analysis and AI-powered trading signals
          </motion.p>
        </div>

        <motion.div
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="flex items-center space-x-4"
        >
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRefresh}
            className="btn-primary flex items-center"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRunCycle}
            className="btn-secondary flex items-center"
          >
            <Play className="w-4 h-4 mr-2" />
            Run Cycle
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="p-3 glass-card hover:bg-white/10 transition-all duration-300"
          >
            <Bell className="w-5 h-5" />
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="p-3 glass-card hover:bg-white/10 transition-all duration-300"
          >
            <User className="w-5 h-5" />
          </motion.button>
        </motion.div>
      </div>
    </motion.header>
  );
}
