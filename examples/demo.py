"""
Demo script for Mismatched Energy Trading System
Shows how to use the system for analysis and backtesting
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MismatchedEnergySystem
import json
from datetime import datetime, timedelta

async def demo_analysis():
    """Demo: Analyze single symbols for mismatched energy"""
    print("🔍 Demo: Single Symbol Analysis")
    print("=" * 50)
    
    system = MismatchedEnergySystem()
    
    # Initialize system
    if not await system.initialize():
        print("❌ Failed to initialize system")
        return
    
    # Analyze specific symbols
    symbols = ['RELIANCE', 'TCS', 'INFY']
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        analysis = await system.analyze_single_symbol(symbol)
        
        if analysis:
            print(f"Symbol: {analysis['symbol']}")
            print(f"Sentiment Score: {analysis['sentiment_summary']['weighted_sentiment']:.3f}")
            print(f"Discord Score: {analysis['mismatch_analysis']['discord_score']:.3f}")
            print(f"Mismatch Detected: {analysis['mismatch_analysis']['is_mismatched']}")
            print(f"Confidence: {analysis['mismatch_analysis']['confidence']:.3f}")
        else:
            print(f"❌ Failed to analyze {symbol}")
    
    await system.shutdown()

async def demo_market_overview():
    """Demo: Get market overview"""
    print("\n🌐 Demo: Market Overview")
    print("=" * 50)
    
    system = MismatchedEnergySystem()
    
    if not await system.initialize():
        print("❌ Failed to initialize system")
        return
    
    overview = await system.get_market_overview()
    
    if overview:
        print(f"Market Analysis at {overview['timestamp']}")
        print(f"Total Symbols: {overview['market_summary']['total_symbols']}")
        print(f"Average Discord Score: {overview['market_summary']['avg_discord_score']:.3f}")
        print(f"Max Discord Score: {overview['market_summary']['max_discord_score']:.3f}")
        print(f"Mismatched Symbols: {overview['market_summary']['mismatched_symbols']}")
        
        print("\n📈 Top Mismatched Opportunities:")
        sorted_symbols = sorted(
            overview['symbols'].items(),
            key=lambda x: x[1].get('mismatch_analysis', {}).get('discord_score', 0),
            reverse=True
        )
        
        for symbol, data in sorted_symbols[:3]:
            discord_score = data.get('mismatch_analysis', {}).get('discord_score', 0)
            if discord_score > 0.1:
                print(f"  {symbol}: Discord Score {discord_score:.3f}")
    
    await system.shutdown()

async def demo_backtest():
    """Demo: Run backtest"""
    print("\n📊 Demo: Backtesting")
    print("=" * 50)
    
    system = MismatchedEnergySystem()
    
    if not await system.initialize():
        print("❌ Failed to initialize system")
        return
    
    # Run backtest for last 30 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    symbols = ['RELIANCE', 'TCS', 'INFY']
    
    print(f"Running backtest from {start_date} to {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    
    results = await system.run_backtest(symbols, start_date, end_date)
    
    if results:
        print(f"\n✅ Backtest Results:")
        print(f"Initial Value: ₹{results['initial_value']:,.2f}")
        print(f"Final Value: ₹{results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {results['num_trades']}")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
    else:
        print("❌ Backtest failed")
    
    await system.shutdown()

async def demo_live_signals():
    """Demo: Get live trading signals"""
    print("\n⚡ Demo: Live Trading Signals")
    print("=" * 50)
    
    system = MismatchedEnergySystem()
    
    if not await system.initialize():
        print("❌ Failed to initialize system")
        return
    
    # Get live signals
    signals = await system.live_executor.get_trading_signals()
    
    if signals:
        print(f"Found {len(signals)} trading signals:")
        for signal in signals:
            print(f"  {signal['symbol']}: Discord {signal['discord_score']:.3f}, "
                  f"Confidence {signal['confidence']:.3f}, "
                  f"Price ₹{signal['current_price']:.2f}")
    else:
        print("No trading signals found")
    
    await system.shutdown()

async def main():
    """Run all demos"""
    print("🚀 Mismatched Energy Trading System - Demo")
    print("=" * 60)
    
    try:
        # Run demos
        await demo_analysis()
        await demo_market_overview()
        await demo_backtest()
        await demo_live_signals()
        
        print("\n✅ All demos completed successfully!")
        print("\n💡 Next Steps:")
        print("1. Configure your Fyers API credentials in .env")
        print("2. Run backtest with real data: python main.py --mode backtest")
        print("3. Start paper trading: python main.py --mode live")
        print("4. Monitor performance and adjust parameters")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
