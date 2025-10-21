import React from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  TrendingUp, 
  Settings, 
  Activity,
  Zap,
  Brain,
  Database
} from 'lucide-react';
import { SystemStatus } from './SystemStatus';

const navItems = [
  { icon: BarChart3, label: 'Dashboard', active: true },
  { icon: TrendingUp, label: 'Trading', active: false },
  { icon: Activity, label: 'Analytics', active: false },
  { icon: Brain, label: 'AI Signals', active: false },
  { icon: Settings, label: 'Settings', active: false },
];

export function Sidebar() {
  return (
    <motion.div
      initial={{ x: -300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="w-72 glass-card m-4 flex flex-col"
    >
      {/* Logo */}
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="p-6 flex items-center"
      >
        <div className="w-10 h-10 bg-gradient-to-br from-primary-500 via-purple-500 to-pink-500 rounded-xl flex items-center justify-center mr-3 glow-effect">
          <Zap className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-gradient">S.E.N.T.R.Y</h1>
          <p className="text-xs text-gray-400">Trading Intelligence</p>
        </div>
      </motion.div>

      {/* Navigation */}
      <nav className="flex-1 px-4">
        {navItems.map((item, index) => (
          <motion.div
            key={item.label}
            initial={{ x: -50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.3 + index * 0.1, duration: 0.4 }}
            className={`mb-2 p-3 rounded-xl cursor-pointer transition-all duration-300 ${
              item.active
                ? 'bg-primary-500/20 border border-primary-500/30 text-primary-400'
                : 'hover:bg-white/5 text-gray-300 hover:text-white'
            }`}
          >
            <div className="flex items-center">
              <item.icon className="w-5 h-5 mr-3" />
              <span className="font-medium">{item.label}</span>
            </div>
          </motion.div>
        ))}
      </nav>

      {/* System Status */}
      <motion.div
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.8, duration: 0.5 }}
        className="p-4"
      >
        <SystemStatus />
      </motion.div>
    </motion.div>
  );
}
