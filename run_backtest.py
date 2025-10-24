"""
Run realistic backtest with full simulation
"""
import asyncio
import argparse
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd

from backtesting import RealisticBacktest, BacktestConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_results(results: dict):
    """Plot backtest results"""
    
    # Extract data
    snapshots = results['snapshots']
    timestamps = [datetime.fromtimestamp(s.timestamp) for s in snapshots]
    values = [s.total_value for s in snapshots]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Portfolio Value
    axes[0].plot(timestamps, values, linewidth=2, color='blue')
    axes[0].axhline(y=results['initial_capital'], color='gray', linestyle='--', label='Initial Capital')
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value (₹)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add return annotation
    total_return = results['total_return_pct']
    axes[0].text(0.02, 0.98, f'Return: {total_return:+.2f}%', 
                transform=axes[0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=12, fontweight='bold')
    
    # 2. Drawdown
    peak = pd.Series(values).expanding().max()
    drawdown = (pd.Series(values) - peak) / peak * 100
    
    axes[1].fill_between(timestamps, drawdown, 0, alpha=0.3, color='red')
    axes[1].plot(timestamps, drawdown, linewidth=1, color='darkred')
    axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add max drawdown annotation
    max_dd = results['max_drawdown'] * 100
    axes[1].text(0.02, 0.02, f'Max Drawdown: {max_dd:.2f}%', 
                transform=axes[1].transAxes,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                fontsize=12, fontweight='bold')
    
    # 3. Number of Positions
    num_positions = [s.num_positions for s in snapshots]
    axes[2].plot(timestamps, num_positions, linewidth=2, color='green')
    axes[2].set_title('Number of Open Positions', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylabel('Positions', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    logger.info(f"Results plot saved to {filename}")
    plt.close()


def print_results(results: dict):
    """Print detailed backtest results"""
    
    print("\n" + "="*80)
    print(" REALISTIC BACKTEST RESULTS")
    print("="*80)
    
    print(f"\n📅 Period: {results['start_date']} to {results['end_date']}")
    print(f"💰 Initial Capital: ₹{results['initial_capital']:,.2f}")
    print(f"💰 Final Capital: ₹{results['final_capital']:,.2f}")
    
    print(f"\n📈 PERFORMANCE METRICS")
    print("-" * 80)
    print(f"Total Return: {results['total_return_pct']:+.2f}%")
    print(f"Total P&L: ₹{results['total_pnl']:+,.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    
    print(f"\n📊 TRADING STATISTICS")
    print("-" * 80)
    print(f"Total Trades: {results['num_trades']}")
    print(f"Closed Positions: {results['num_closed_positions']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Average Win: ₹{results['avg_win']:,.2f}")
    print(f"Average Loss: ₹{results['avg_loss']:,.2f}")
    
    if results['avg_loss'] != 0:
        profit_factor = abs(results['avg_win'] / results['avg_loss'])
        print(f"Profit Factor: {profit_factor:.2f}")
    
    print(f"\n💸 COSTS")
    print("-" * 80)
    print(f"Total Commissions: ₹{results['total_commissions']:,.2f}")
    print(f"Commission % of Initial Capital: {results['total_commissions']/results['initial_capital']*100:.3f}%")
    
    print(f"\n🎯 SIGNALS")
    print("-" * 80)
    print(f"Signals Generated: {results['num_signals_generated']}")
    print(f"Signal Conversion Rate: {results['num_trades']/max(results['num_signals_generated'],1)*100:.2f}%")
    
    print("\n" + "="*80)


async def main():
    parser = argparse.ArgumentParser(description='Run realistic backtest')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to backtest')
    parser.add_argument('--save-json', action='store_true', help='Save results to JSON')
    
    args = parser.parse_args()
    
    # Load symbols
    if args.symbols:
        symbols = args.symbols
    else:
        try:
            from data.tickers import get_tickers
            symbols = get_tickers()[:10]  # Test with first 10
            logger.info(f"Using {len(symbols)} symbols from tickers")
        except:
            symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
            logger.warning(f"Could not load tickers, using default: {symbols}")
    
    # Create config
    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        max_positions=5,
        position_size_pct=0.20,  # 20% per position
        min_discord_score=0.3,
        min_confidence=0.6,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        max_holding_days=5
    )
    
    # Run backtest
    logger.info("="*80)
    logger.info("STARTING REALISTIC BACKTEST")
    logger.info("="*80)
    
    backtest = RealisticBacktest(config)
    results = await backtest.run(symbols)
    
    # Print results
    print_results(results)
    
    # Plot results
    try:
        plot_results(results)
    except Exception as e:
        logger.error(f"Could not plot results: {e}")
    
    # Save to JSON
    if args.save_json:
        # Convert non-serializable objects
        results_json = {
            k: v for k, v in results.items() 
            if k not in ['snapshots', 'closed_positions']
        }
        
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        logger.info(f"Results saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())
