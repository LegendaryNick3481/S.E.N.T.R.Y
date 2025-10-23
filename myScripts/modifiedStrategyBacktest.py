import pandas as pd
import numpy as np
from datetime import time, datetime
from decimal import Decimal, getcontext
import credentials as crs
from fyers_apiv3 import fyersModel


class RenkoBacktester:
    def __init__(self, symbol, brick_size, qty, prev_day_renko_close, datefrom, dateto, slippage=0.1):
        self.symbol = symbol
        self.brick_size = brick_size
        self.qty = qty
        self.init_price = prev_day_renko_close
        self.trades = []
        self.datefrom = datefrom
        self.dateto = dateto
        self.position = None
        self.trades = []
        self.indicators_history = []
        self.current_day_bricks = None

        # Slippage factor
        self.slippage = slippage

        # Continuous brick storage for indicator calculations
        self.all_bricks_continuous = pd.DataFrame()

        # Trading parameters - Removed choppiness parameters
        self.momentum_period = 7

        # Square off time (3:15 PM)
        self.square_off_time = time(15, 15)

        # For fetching data from socket
        self.access_token = ""
        self.client_id = crs.client_id
        with open('bufferFiles/access_token.txt') as file: self.access_token = file.read().strip()
        self.fyers = fyersModel.FyersModel(client_id=self.client_id, is_async=False, token=self.access_token,
                                           log_path="bufferFiles/")

        # File generation
        self.fetchData('bufferFiles/backtestdata.csv')

    def apply_slippage(self, price, is_buy):
        """
        Apply slippage to the execution price

        Args:
            price: Original price from brick close
            is_buy: True for buy orders (entry), False for sell orders (exit)

        Returns:
            Adjusted price with slippage
        """
        if is_buy:
            # For buy orders, we pay slippage (higher price)
            return price + self.slippage
        else:
            # For sell orders, we lose slippage (lower price)
            return price - self.slippage

    def fetchData(self, _filename):
        filename = _filename

        data = {
            "symbol": self.symbol,
            "resolution": "1",
            "date_format": "1",
            "range_from": self.datefrom,
            "range_to": self.dateto,
            "cont_flag": "1"
        }

        sdata = self.fyers.history(data=data)
        for i in range(len(sdata["candles"])):
            sdata["candles"][i][0] = datetime.fromtimestamp(sdata["candles"][i][0])
        df = pd.DataFrame(sdata["candles"])
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df.to_csv(filename, index=False)

    def calculate_charges(self, buy_value, sell_value):
        """Calculate intraday trading charges"""
        turnover = buy_value + sell_value
        brokerage = min(40, 0.0003 * turnover)  # 0.03% or ₹20 per leg
        stt = 0.00025 * sell_value
        transaction = turnover * 0.0000325
        gst = 0.18 * (brokerage + transaction)
        sebi = turnover * 0.000001
        stamp = 0.00003 * buy_value
        return brokerage + stt + transaction + gst + sebi + stamp

    def generate_renko(self, df):
        """Generate Renko bricks from 1-minute candles (precision-fixed version)"""
        getcontext().prec = 10
        bricks = []

        last_close = Decimal(str(self.init_price))
        brick_size = Decimal(str(self.brick_size))
        last_dir = None

        for _, row in df.iterrows():
            ts = row['timestamp']
            price = Decimal(str(row['close']))

            while True:
                diff = price - last_close

                if abs(diff) < brick_size:
                    break

                if last_dir is None:
                    direction = 1 if diff > 0 else 0
                    new_close = last_close + (brick_size if direction else -brick_size)
                    bricks.append({
                        'timestamp': ts,
                        'open': float(last_close),
                        'high': float(max(last_close, new_close)),
                        'low': float(min(last_close, new_close)),
                        'close': float(new_close),
                        'direction': direction
                    })
                    last_close = new_close
                    last_dir = direction
                    continue

                if last_dir == 1:
                    if diff >= brick_size:
                        new_close = last_close + brick_size
                        bricks.append({
                            'timestamp': ts,
                            'open': float(last_close),
                            'high': float(new_close),
                            'low': float(last_close),
                            'close': float(new_close),
                            'direction': 1
                        })
                        last_close = new_close
                    elif diff <= -2 * brick_size:
                        new_close = last_close - 2 * brick_size
                        bricks.append({
                            'timestamp': ts,
                            'open': float(last_close - brick_size),
                            'high': float(last_close - brick_size),
                            'low': float(new_close),
                            'close': float(new_close),
                            'direction': 0
                        })
                        last_close = new_close
                        last_dir = 0
                    else:
                        break

                elif last_dir == 0:
                    if diff <= -brick_size:
                        new_close = last_close - brick_size
                        bricks.append({
                            'timestamp': ts,
                            'open': float(last_close),
                            'high': float(last_close),
                            'low': float(new_close),
                            'close': float(new_close),
                            'direction': 0
                        })
                        last_close = new_close
                    elif diff >= 2 * brick_size:
                        new_close = last_close + 2 * brick_size
                        bricks.append({
                            'timestamp': ts,
                            'open': float(last_close + brick_size),
                            'high': float(new_close),
                            'low': float(last_close + brick_size),
                            'close': float(new_close),
                            'direction': 1
                        })
                        last_close = new_close
                        last_dir = 1
                    else:
                        break

        return pd.DataFrame(bricks).reset_index(drop=True)

    def calculate_indicators_continuous(self, brick_index):
        """Calculate momentum using continuous brick history - Removed choppiness"""
        if brick_index < self.momentum_period:
            return 1

        # Momentum calculation
        closes = self.all_bricks_continuous['close'].values
        momentum_start_idx = max(0, brick_index - self.momentum_period)
        if momentum_start_idx < len(closes):
            momentum = closes[brick_index] - closes[momentum_start_idx]
        else:
            momentum = 1

        return momentum

    def open_position(self, ts, theoretical_price, momentum):
        """Record new long position with slippage applied"""
        actual_entry_price = self.apply_slippage(theoretical_price, is_buy=True)

        self.position = 'LONG'
        self.entry_price = actual_entry_price
        self.theoretical_entry_price = theoretical_price
        self.entry_time = ts
        self.entry_momentum = momentum

    def close_position(self, ts, theoretical_price, reason):
        """Close existing position with slippage applied"""
        if self.position != 'LONG':
            return

        actual_exit_price = self.apply_slippage(theoretical_price, is_buy=False)
        exit_time = ts

        # Calculate P&L using actual executed prices (with slippage)
        gross_pnl = (actual_exit_price - self.entry_price) * self.qty
        charges = self.calculate_charges(self.entry_price * self.qty, actual_exit_price * self.qty)
        net_pnl = gross_pnl - charges

        # Calculate slippage impact correctly
        # Entry slippage cost: We paid MORE than theoretical (always negative impact)
        entry_slippage_cost = (self.theoretical_entry_price - self.entry_price) * self.qty  # Should be negative

        # Exit slippage cost: We received LESS than theoretical (always negative impact)
        exit_slippage_cost = (actual_exit_price - theoretical_price) * self.qty  # Should be negative

        # Total slippage impact (sum of both costs - should be negative)
        total_slippage_impact = entry_slippage_cost + exit_slippage_cost

        self.trades.append({
            'symbol': self.symbol,
            'entry_time': self.entry_time,
            'exit_time': exit_time,
            'entry_price': self.entry_price,
            'exit_price': actual_exit_price,
            'theoretical_entry_price': self.theoretical_entry_price,
            'theoretical_exit_price': theoretical_price,
            'qty': self.qty,
            'gross_pnl': gross_pnl,
            'charges': charges,
            'net_pnl': net_pnl,
            'slippage_impact': total_slippage_impact,
            'entry_slippage': entry_slippage_cost,
            'exit_slippage': exit_slippage_cost,
            'reason': reason,
            'entry_momentum': self.entry_momentum
        })

        self.position = None
        self.entry_price = 0
        self.theoretical_entry_price = 0

    def get_closest_price_at_time(self, df, target_time):
        """Get the closest available price at or after the target time"""
        df_time_filtered = df[df['timestamp'].dt.time >= target_time]
        if df_time_filtered.empty:
            return df.iloc[-1]['close'], df.iloc[-1]['timestamp']
        else:
            first_row = df_time_filtered.iloc[0]
            return first_row['close'], first_row['timestamp']

    def identify_last_bricks_per_day(self, bricks_df):
        """
        Identify the last brick for each trading day
        Returns a set of indices that represent last bricks of each day
        """
        last_brick_indices = set()

        if bricks_df.empty:
            return last_brick_indices

        bricks_df['date'] = bricks_df['timestamp'].dt.date

        for date, group in bricks_df.groupby('date'):
            last_idx = group.index[-1]
            last_brick_indices.add(last_idx)

        return last_brick_indices

    def calculate_daily_pnl(self, trades_df):
        """Calculate per-day P&L from trades"""
        if trades_df.empty:
            return pd.DataFrame()

        trades_df = trades_df.copy()
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df['trade_date'] = trades_df['exit_time'].dt.date

        daily_pnl = trades_df.groupby('trade_date').agg({
            'net_pnl': 'sum',
            'gross_pnl': 'sum',
            'charges': 'sum',
            'slippage_impact': 'sum',
            'symbol': 'count'
        }).rename(columns={'symbol': 'trade_count'})

        daily_pnl['cumulative_pnl'] = daily_pnl['net_pnl'].cumsum()
        daily_pnl = daily_pnl.reset_index()

        return daily_pnl

    def backtest(self, df):
        """Run backtest without choppiness - simplified logic"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        print("Generating continuous Renko bricks across all days...")
        self.all_bricks_continuous = self.generate_renko(df)
        print(f"Total bricks generated: {len(self.all_bricks_continuous)}")
        print(f"Slippage factor: {self.slippage} points per trade leg")

        self.all_bricks_continuous = self.all_bricks_continuous.reset_index(drop=True)

        unique_dates = sorted(df['timestamp'].dt.date.unique())

        for day_idx in range(1, len(unique_dates)):
            current_date = unique_dates[day_idx]
            prev_date = unique_dates[day_idx - 1]

            print(f"\nProcessing {current_date} (Previous day: {prev_date})")

            current_day_df = df[df['timestamp'].dt.date == current_date]

            if current_day_df.empty:
                print(f"Skipping {current_date} due to insufficient data")
                continue

            current_day_bricks = self.all_bricks_continuous[
                self.all_bricks_continuous['timestamp'].dt.date == current_date
                ].copy()

            if current_day_bricks.empty:
                print(f"No bricks generated for {current_date}")
                continue

            self.current_day_bricks = current_day_bricks.reset_index(drop=True)
            current_day_continuous_indices = current_day_bricks.index.tolist()

            current_day_bricks_reset = current_day_bricks.reset_index(drop=True)
            last_brick_indices = self.identify_last_bricks_per_day(current_day_bricks_reset)
            print(f"Last brick indices to avoid trading: {last_brick_indices}")

            chart_pattern = []

            for local_idx, (continuous_idx, brick) in enumerate(current_day_bricks.iterrows()):
                ts = brick['timestamp']
                direction = brick['direction']
                theoretical_price = brick['close']

                is_last_brick_of_day = local_idx in last_brick_indices

                momentum = self.calculate_indicators_continuous(continuous_idx)

                self.indicators_history.append({
                    'timestamp': ts,
                    'momentum': momentum,
                    'continuous_index': continuous_idx,
                    'local_index': local_idx
                })

                # Mandatory square off at 3:15 PM
                if ts.time() >= self.square_off_time and self.position == 'LONG':
                    self.close_position(ts, theoretical_price, 'EOD')
                    print(f"MANDATORY square off at 3:15 PM: {ts} - Theoretical Price: {theoretical_price}")
                    continue

                if ts.time() >= self.square_off_time:
                    continue

                if time(9, 15) <= ts.time() < time(9, 16):
                    continue

                if is_last_brick_of_day:
                    print(f"SKIPPING trade decision on LAST BRICK of day at {ts}")
                    continue

                chart_pattern = (chart_pattern + [direction])[-3:]  # Keep last 3 for [1,1,1] pattern

                # Debug print to see what patterns we're getting
                if len(chart_pattern) >= 2:
                    print(f"Pattern at {ts}: {chart_pattern}, Direction: {direction}, Momentum: {momentum:.2f}")

                # Buy logic - Only buy at pattern [0,1] with positive momentum
                if self.position is None and len(chart_pattern) >= 2:
                    last_two = chart_pattern[-2:]  # Get last 2 elements
                    if last_two in  ([0, 1],[1,1]):
                        print(f"Checking buy conditions: Pattern {last_two}, Momentum {momentum:.2f}")
                        if momentum > 0:
                            actual_entry_price = self.apply_slippage(theoretical_price, is_buy=True)
                            print(f"Opening LONG position at {ts} - "
                                  f"Theoretical Price: {theoretical_price}, Actual Price: {actual_entry_price:.2f} "
                                  f"(Slippage: +{self.slippage}) - Pattern: [0,1]")
                            self.open_position(ts, theoretical_price, momentum)
                        else:
                            print(f"Buy condition failed: Momentum not positive ({momentum:.2f})")

                # Replace the existing sell logic section in your backtest method with this:

                # Sell logic - Exit on red brick OR after 2 Rs profit target
                elif self.position == 'LONG':
                    print(f"In position - checking exit conditions. Current pattern: {chart_pattern}")

                    # Calculate profit target (2 * brick_size from theoretical entry price)
                    profit_target = self.theoretical_entry_price + (2 * self.brick_size)
                    current_profit = theoretical_price - self.theoretical_entry_price

                    # Exit condition 1: Red brick (stop loss)
                    if len(chart_pattern) >= 1 and chart_pattern[-1] == 0:
                        actual_exit_price = self.apply_slippage(theoretical_price, is_buy=False)
                        print(f"Closing position due to RED_BRICK at {ts} - "
                              f"Theoretical Price: {theoretical_price}, Actual Price: {actual_exit_price:.2f} "
                              f"(Slippage: -{self.slippage}) - Pattern ends with 0")
                        self.close_position(ts, theoretical_price, 'RED_BRICK')

                    # Exit condition 2: 2 Rs profit target reached
                    elif theoretical_price >= profit_target:
                        actual_exit_price = self.apply_slippage(theoretical_price, is_buy=False)
                        print(f"Closing position due to PROFIT_TARGET at {ts} - "
                              f"Theoretical Price: {theoretical_price}, Actual Price: {actual_exit_price:.2f} "
                              f"(Slippage: -{self.slippage}) - Profit: {current_profit:.2f} Rs (Target: 2.00 Rs)")
                        self.close_position(ts, theoretical_price, 'PROFIT_TARGET')

                    else:
                        print(f"No exit condition met - Pattern: {chart_pattern}, "
                              f"Current Profit: {current_profit:.2f} Rs, Target: 2.00 Rs")

            # Ensure all positions are squared off by 3:15 PM
            if self.position == 'LONG':
                try:
                    square_off_price, square_off_time = self.get_closest_price_at_time(current_day_df,
                                                                                       self.square_off_time)
                    self.close_position(square_off_time, square_off_price, 'EOD')
                    print(f"END OF DAY square off executed at {square_off_time}")
                except Exception as e:
                    last_price = current_day_df.iloc[-1]['close']
                    last_time = current_day_df.iloc[-1]['timestamp']
                    self.close_position(last_time, last_price, 'EMERGENCY_SQUARE_OFF')
                    print(f"EMERGENCY square off executed at {last_time}")
                    print(f"ERROR during square off: {e}")

        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame(columns=[
            'symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'theoretical_entry_price', 'theoretical_exit_price', 'qty', 'gross_pnl',
            'charges', 'net_pnl', 'slippage_impact', 'entry_slippage', 'exit_slippage',
            'reason', 'entry_momentum'
        ])


if __name__ == "__main__":
    data = pd.read_csv("bufferFiles/backtestdata.csv")
    backtester = RenkoBacktester(
        symbol="NSE:GARUDA-EQ",
        brick_size=1,
        qty=22,
        prev_day_renko_close=114.00,
        datefrom='2025-06-02',
        dateto='2025-06-20',
        slippage=0.1
    )

    results = backtester.backtest(data)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    if not results.empty:
        print("\nTrade Results:")
        print(results[['entry_time', 'exit_time', 'theoretical_entry_price', 'theoretical_exit_price',
                       'entry_momentum', 'net_pnl', 'reason']])

        total_net = results['net_pnl'].sum()
        total_slippage_impact = results['slippage_impact'].sum()
        total_entry_slippage = results['entry_slippage'].sum()
        total_exit_slippage = results['exit_slippage'].sum()
        theoretical_pnl = total_net - total_slippage_impact

        print(f"\nPerformance Summary:")
        print(f"Total Net P&L (with slippage): {total_net:.2f}")
        print(f"Total Slippage Impact: {total_slippage_impact:.2f}")
        print(f"  - Entry Slippage: {total_entry_slippage:.2f}")
        print(f"  - Exit Slippage: {total_exit_slippage:.2f}")
        print(f"Theoretical P&L (without slippage): {theoretical_pnl:.2f}")
        print(f"Number of Trades: {len(results)}")
        print(f"Average P&L per Trade: {total_net / len(results) if len(results) > 0 else 0:.2f}")
        print(
            f"Average Slippage Impact per Trade: {total_slippage_impact / len(results) if len(results) > 0 else 0:.2f}")

        # Show breakdown by exit reasons
        print(f"\nExit Reason Breakdown:")
        exit_reasons = results['reason'].value_counts()
        for reason, count in exit_reasons.items():
            reason_pnl = results[results['reason'] == reason]['net_pnl'].sum()
            reason_slippage = results[results['reason'] == reason]['slippage_impact'].sum()
            print(f"{reason}: {count} trades, Net P&L: {reason_pnl:.2f}, Slippage Impact: {reason_slippage:.2f}")

        # Calculate and display daily P&L
        daily_pnl = backtester.calculate_daily_pnl(results)
        if not daily_pnl.empty:
            print(f"\n{'=' * 75}")
            print("DAILY P&L BREAKDOWN:")
            print(f"{'=' * 75}")
            print(
                f"{'Date':<12} {'Trades':<7} {'Gross P&L':<10} {'Charges':<8} {'Slippage':<9} {'Net P&L':<10} {'Cumulative':<12}")
            print(f"{'-' * 75}")
            for _, row in daily_pnl.iterrows():
                print(f"{str(row['trade_date']):<12} {int(row['trade_count']):<7} "
                      f"{row['gross_pnl']:>9.2f} {row['charges']:>7.2f} "
                      f"{row['slippage_impact']:>8.2f} {row['net_pnl']:>9.2f} {row['cumulative_pnl']:>11.2f}")

            # Summary statistics
            profitable_days = len(daily_pnl[daily_pnl['net_pnl'] > 0])
            loss_days = len(daily_pnl[daily_pnl['net_pnl'] < 0])
            break_even_days = len(daily_pnl[daily_pnl['net_pnl'] == 0])

            print(f"\n{'=' * 75}")
            print("DAILY P&L SUMMARY:")
            print(f"{'=' * 75}")
            print(f"Total Trading Days: {len(daily_pnl)}")
            print(f"Profitable Days: {profitable_days} ({profitable_days / len(daily_pnl) * 100:.1f}%)")
            print(f"Loss Days: {loss_days} ({loss_days / len(daily_pnl) * 100:.1f}%)")
            print(f"Break-even Days: {break_even_days} ({break_even_days / len(daily_pnl) * 100:.1f}%)")

            if profitable_days > 0:
                avg_profit = daily_pnl[daily_pnl['net_pnl'] > 0]['net_pnl'].mean()
                max_profit = daily_pnl['net_pnl'].max()
                print(f"Average Profit Day: {avg_profit:.2f}")
                print(f"Best Day: {max_profit:.2f}")

            if loss_days > 0:
                avg_loss = daily_pnl[daily_pnl['net_pnl'] < 0]['net_pnl'].mean()
                max_loss = daily_pnl['net_pnl'].min()
                print(f"Average Loss Day: {avg_loss:.2f}")
                print(f"Worst Day: {max_loss:.2f}")

            print(f"Average Daily P&L: {daily_pnl['net_pnl'].mean():.2f}")
            print(f"Average Daily Slippage Impact: {daily_pnl['slippage_impact'].mean():.2f}")
            print(f"Daily P&L Std Dev: {daily_pnl['net_pnl'].std():.2f}")

    else:
        print("No trades were executed during the backtest period.")

    # Print sample indicators for verification
    print("\nSample Computed Indicators:")
    indicators_df = pd.DataFrame(backtester.indicators_history)
    if not indicators_df.empty:
        print(indicators_df.head(10))

    # Print sample bricks for verification
    print("\nSample Renko Bricks:")
    if backtester.current_day_bricks is not None:
        print(backtester.current_day_bricks.head(10))