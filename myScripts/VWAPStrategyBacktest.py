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

        # Slippage factor
        self.slippage = slippage

        # Continuous brick storage for VWAP calculations
        self.all_bricks_continuous = pd.DataFrame()

        # Square off time (3:15 PM)
        self.square_off_time = time(15, 5)

        # For fetching data from socket
        self.access_token = ""
        self.client_id = crs.client_id
        with open('bufferFiles/access_token.txt') as file:
            self.access_token = file.read().strip()
        self.fyers = fyersModel.FyersModel(client_id=self.client_id, is_async=False, token=self.access_token,
                                           log_path="bufferFiles/")

        # File generation
        self.fetchData('bufferFiles/backtestdata.csv')

    def apply_slippage(self, price, is_buy):
        """Apply slippage to the execution price"""
        if is_buy:
            return price + self.slippage
        else:
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
        brokerage = min(40, 0.0003 * turnover)
        stt = 0.00025 * sell_value
        transaction = turnover * 0.0000325
        gst = 0.18 * (brokerage + transaction)
        sebi = turnover * 0.000001
        stamp = 0.00003 * buy_value
        return brokerage + stt + transaction + gst + sebi + stamp

    def generate_renko(self, df):
        """Generate Renko bricks from 3-minute candles with volume"""
        getcontext().prec = 10
        bricks = []

        last_close = Decimal(str(self.init_price))
        brick_size = Decimal(str(self.brick_size))
        last_dir = None

        for _, row in df.iterrows():
            ts = row['timestamp']
            price = Decimal(str(row['close']))
            volume = row['volume']

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
                        'direction': direction,
                        'volume': volume
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
                            'direction': 1,
                            'volume': volume
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
                            'direction': 0,
                            'volume': volume
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
                            'direction': 0,
                            'volume': volume
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
                            'direction': 1,
                            'volume': volume
                        })
                        last_close = new_close
                        last_dir = 1
                    else:
                        break

        return pd.DataFrame(bricks).reset_index(drop=True)

    def calculate_vwap(self, bricks_df, current_index):
        """Calculate VWAP based on brick close and volume for current day"""
        if current_index < 0 or bricks_df.empty:
            return 0

        current_brick = bricks_df.iloc[current_index]
        current_date = current_brick['timestamp'].date()

        # Get all bricks for current day up to current index
        day_bricks = bricks_df[
            (bricks_df['timestamp'].dt.date == current_date) &
            (bricks_df.index <= current_index)
            ].copy()

        if day_bricks.empty:
            return 0

        # Calculate VWAP using brick close and volume
        day_bricks['price_volume'] = day_bricks['close'] * day_bricks['volume']
        total_price_volume = day_bricks['price_volume'].sum()
        total_volume = day_bricks['volume'].sum()

        if total_volume == 0:
            return day_bricks['close'].iloc[-1]  # Fallback to last close if no volume

        vwap = total_price_volume / total_volume
        return vwap

    def open_position(self, ts, theoretical_price, vwap):
        """Record new long position with slippage applied"""
        actual_entry_price = self.apply_slippage(theoretical_price, is_buy=True)

        self.position = 'LONG'
        self.entry_price = actual_entry_price
        self.theoretical_entry_price = theoretical_price
        self.entry_time = ts
        self.entry_vwap = round(vwap, 2)

    def close_position(self, ts, theoretical_price, reason, vwap):
        """Close existing position with slippage applied"""
        if self.position != 'LONG':
            return

        actual_exit_price = self.apply_slippage(theoretical_price, is_buy=False)
        exit_time = ts

        # Calculate P&L using actual executed prices (with slippage)
        gross_pnl = (actual_exit_price - self.entry_price) * self.qty
        charges = self.calculate_charges(self.entry_price * self.qty, actual_exit_price * self.qty)
        net_pnl = gross_pnl - charges

        # Calculate slippage impact
        entry_slippage_cost = (self.entry_price - self.theoretical_entry_price) * self.qty
        exit_slippage_cost = (actual_exit_price - theoretical_price) * self.qty
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
            'entry_vwap': self.entry_vwap,
            'exit_vwap': round(vwap, 2)
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
        """Run backtest with pattern + VWAP entry logic and red brick exit"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        print("Generating continuous Renko bricks from 3-minute candles across all days...")
        self.all_bricks_continuous = self.generate_renko(df)
        print(f"Total bricks generated: {len(self.all_bricks_continuous)}")
        print(f"Slippage factor: {self.slippage} points per trade leg")
        print("Strategy: Buy when ([0,1] or [1,1]) pattern AND price > VWAP, Sell on red brick")

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

            chart_pattern = []

            for idx, brick in current_day_bricks.iterrows():
                ts = brick['timestamp']
                direction = brick['direction']
                theoretical_price = brick['close']

                if ts.time() <= time(9, 15):
                    continue

                # Mandatory square off at 3:15 PM
                if ts.time() >= self.square_off_time and self.position == 'LONG':
                    vwap = self.calculate_vwap(self.all_bricks_continuous, idx)
                    self.close_position(ts, theoretical_price, 'EOD', vwap)
                    print(f"MANDATORY square off at 3:15 PM: {ts} - Price: {theoretical_price:.2f}, VWAP: {vwap:.2f}")
                    continue

                if ts.time() >= self.square_off_time:
                    continue

                # Calculate VWAP for current brick
                vwap = self.calculate_vwap(self.all_bricks_continuous, idx)

                # Update chart pattern (keep last 2 directions)
                chart_pattern = (chart_pattern + [direction])[-2:]

                # Buy logic: ([0,1] or [1,1]) pattern AND price > VWAP and no existing position
                if self.position is None and len(chart_pattern) == 2:
                    if chart_pattern in ([0, 1], [1, 1]) and theoretical_price > vwap:
                        actual_entry_price = self.apply_slippage(theoretical_price, is_buy=True)
                        print(f"Opening LONG position at {ts} - "
                              f"Pattern: {chart_pattern}, Price: {theoretical_price:.2f} > VWAP: {vwap:.2f}, "
                              f"Actual Price: {actual_entry_price:.2f} (Slippage: +{self.slippage})")
                        self.open_position(ts, theoretical_price, vwap)

                # Sell logic: red brick (direction 0)
                elif self.position == 'LONG' and direction == 0:
                    actual_exit_price = self.apply_slippage(theoretical_price, is_buy=False)
                    print(f"Closing position on RED BRICK at {ts} - "
                          f"Price: {theoretical_price:.2f}, VWAP: {vwap:.2f}, "
                          f"Actual Price: {actual_exit_price:.2f} (Slippage: -{self.slippage})")
                    self.close_position(ts, theoretical_price, 'RED_BRICK', vwap)

            # Ensure all positions are squared off by 3:15 PM
            if self.position == 'LONG':
                try:
                    square_off_price, square_off_time = self.get_closest_price_at_time(current_day_df,
                                                                                       self.square_off_time)
                    last_idx = current_day_bricks.index[-1] if not current_day_bricks.empty else 0
                    vwap = self.calculate_vwap(self.all_bricks_continuous, last_idx)
                    self.close_position(square_off_time, square_off_price, 'EOD', vwap)
                    print(f"END OF DAY square off executed at {square_off_time}")
                except Exception as e:
                    last_price = current_day_df.iloc[-1]['close']
                    last_time = current_day_df.iloc[-1]['timestamp']
                    last_idx = current_day_bricks.index[-1] if not current_day_bricks.empty else 0
                    vwap = self.calculate_vwap(self.all_bricks_continuous, last_idx)
                    self.close_position(last_time, last_price, 'EMERGENCY_SQUARE_OFF', vwap)
                    print(f"EMERGENCY square off executed at {last_time}")
                    print(f"ERROR during square off: {e}")

        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame(columns=[
            'symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'theoretical_entry_price', 'theoretical_exit_price', 'qty', 'gross_pnl',
            'charges', 'net_pnl', 'slippage_impact', 'entry_slippage', 'exit_slippage',
            'reason', 'entry_vwap', 'exit_vwap'
        ])


if __name__ == "__main__":
    data = pd.read_csv("bufferFiles/backtestdata.csv")
    backtester = RenkoBacktester(
        symbol="NSE:TATAMOTORS-EQ",
        brick_size=1,
        qty=4,
        prev_day_renko_close=719.0,
        datefrom='2025-06-01',
        dateto='2025-06-29',
        slippage=0.1
    )

    results = backtester.backtest(data)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    if not results.empty:
        print("\nTrade Results:")
        print(results[['entry_time', 'exit_time', 'theoretical_entry_price',
                       'entry_vwap', 'net_pnl', 'reason']])

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
        print(f"Average Slippage Impact per Trade: {total_slippage_impact / len(results) if len(results) > 0 else 0:.2f}")

        # VWAP statistics
        avg_entry_vwap = results['entry_vwap'].mean()
        avg_exit_vwap = results['exit_vwap'].mean()
        print(f"Average Entry VWAP: {avg_entry_vwap:.2f}")
        print(f"Average Exit VWAP: {avg_exit_vwap:.2f}")

        # Show breakdown by exit reasons
        print(f"\nExit Reason Breakdown:")
        reason_summary = results.groupby('reason').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'slippage_impact': 'sum'
        }).round(2)
        reason_summary.columns = ['Trade_Count', 'Total_PnL', 'Avg_PnL', 'Total_Slippage_Impact']
        print(reason_summary)

        # Calculate daily P&L
        daily_pnl = backtester.calculate_daily_pnl(results)
        if not daily_pnl.empty:
            print(f"\nDaily P&L Summary:")
            print(daily_pnl[['trade_date', 'trade_count', 'net_pnl', 'cumulative_pnl']])

            # Calculate performance metrics
            winning_trades = len(results[results['net_pnl'] > 0])
            losing_trades = len(results[results['net_pnl'] < 0])
            win_rate = (winning_trades / len(results)) * 100 if len(results) > 0 else 0

            avg_win = results[results['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
            avg_loss = results[results['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0

            print(f"\nPerformance Metrics:")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Winning Trades: {winning_trades}")
            print(f"Losing Trades: {losing_trades}")
            print(f"Average Win: {avg_win:.2f}")
            print(f"Average Loss: {avg_loss:.2f}")

            if avg_loss != 0:
                profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades))
                print(f"Profit Factor: {profit_factor:.2f}")

            # Maximum drawdown calculation
            if not daily_pnl.empty:
                cumulative_pnl = daily_pnl['cumulative_pnl'].values
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = cumulative_pnl - running_max
                max_drawdown = np.min(drawdown)
                print(f"Maximum Drawdown: {max_drawdown:.2f}")

        # Save results to CSV
        results.to_csv('bufferFiles/backtest_results.csv', index=False)
        print(f"\nResults saved to 'bufferFiles/backtest_results.csv'")

    else:
        print("No trades were generated during the backtest period.")
        print("This could be due to:")
        print("- No instances where ([0,1] or [1,1]) pattern AND price > VWAP for entry")
        print("- No red bricks generated after positions were opened")
        print("- Insufficient price movement to generate required brick patterns")