import React from 'react';
import { motion } from 'framer-motion';
import { Wifi, WifiOff, Database, Brain } from 'lucide-react';

interface StatusItem {
  name: string;
  status: 'connected' | 'disconnected';
  icon: React.ComponentType<{ className?: string }>;
}

const statusItems: StatusItem[] = [
  { name: 'FYERS API', status: 'disconnected', icon: Database },
  { name: 'News Feed', status: 'connected', icon: Brain },
  { name: 'Analysis Engine', status: 'connected', icon: Brain },
];

export function SystemStatus() {
  return (
    <div className="glass-card p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">System Status</h3>
      <div className="space-y-3">
        {statusItems.map((item, index) => (
          <motion.div
            key={item.name}
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: index * 0.1, duration: 0.3 }}
            className="flex items-center justify-between"
          >
            <div className="flex items-center">
              {item.status === 'connected' ? (
                <Wifi className="w-4 h-4 text-success mr-2" />
              ) : (
                <WifiOff className="w-4 h-4 text-danger mr-2" />
              )}
              <span className="text-sm text-gray-300">{item.name}</span>
            </div>
            <div className={`status-indicator ${
              item.status === 'connected' ? 'status-connected' : 'status-disconnected'
            }`} />
          </motion.div>
        ))}
      </div>
    </div>
  );
}
