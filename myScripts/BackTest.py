"""
Backtest code to test our strategy for a given trading day (need t-day & t-1 day)
Input variables:
1) datefrom ---> t-1 day (The trading day before the 'test' trading day)
2) dateto   ---> t-day (The test trading day)
3) bricksize ---> choose wisely
4) tickername
5) renkoopening of 'dayfrom' ---> obtain manually
6) quantity
"""
import pandas as pd
import numpy as np
from datetime import time, datetime
from decimal import Decimal, getcontext
import credentials as crs
from fyers_apiv3 import fyersModel

class RenkoBacktester:
    def __init__(self, symbol, brick_size, qty, prev_day_renko_close,datefrom,dateto):
        self.symbol = symbol
        self.brick_size = brick_size
        self.qty = qty
        self.init_price = prev_day_renko_close
        self.trades = []
        self.datefrom = datefrom
        self.dateto = dateto
        self.position = None
        self.trades = []
        self.indicators_history = []  # To store all computed indicators
        self.current_day_bricks = None  # To store current day's bricks

        # Trading parameters
        self.choppy_period = 14
        self.momentum_period = 7
        self.choppy_low = 38
        self.choppy_high = 100

        # For fetching data from socket
        self.access_token = ""
        self.client_id = crs.client_id
        with open('bufferFiles/access_token.txt') as file: self.access_token = file.read().strip()
        self.fyers = fyersModel.FyersModel(client_id=self.client_id, is_async=False, token=self.access_token,
                                           log_path="bufferFiles/")

        # File generation
        self.fetchData('bufferFiles/backtestdata.csv')

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
        getcontext().prec = 10  # Set desired precision
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

    def calculate_indicators(self, bricks):
        """Calculate choppiness and momentum for current brick"""
        if len(bricks) < max(self.choppy_period, self.momentum_period) + 1:
            return 0, 1  # Default values

        # Choppiness calculation
        recent = bricks.iloc[-self.choppy_period - 1:]
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values

        tr = np.maximum.reduce([
            highs[1:] - lows[1:],
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        ])

        sum_tr = np.sum(tr)
        max_high = np.max(highs)
        min_low = np.min(lows)
        range_ = max_high - min_low

        choppy = 100 * np.log10(sum_tr / range_) / np.log10(self.choppy_period) if range_ > 0 else 0

        # Momentum calculation
        momentum = closes[-1] - closes[-self.momentum_period - 1]

        return choppy, momentum

    def open_position(self, ts, price, choppy, momentum):
        """Record new long position"""
        self.position = 'LONG'
        self.entry_price = price
        self.entry_time = ts
        self.entry_choppy = choppy
        self.entry_momentum = momentum

    def close_position(self, ts, price, reason, choppy):
        """Close existing position"""
        if self.position != 'LONG':
            return

        exit_price = price
        exit_time = ts

        gross_pnl = (exit_price - self.entry_price) * self.qty
        charges = self.calculate_charges(self.entry_price * self.qty, exit_price * self.qty)
        net_pnl = gross_pnl - charges

        self.trades.append({
            'symbol': self.symbol,
            'entry_time': self.entry_time,
            'exit_time': exit_time,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'qty': self.qty,
            'gross_pnl': gross_pnl,
            'charges': charges,
            'net_pnl': net_pnl,
            'reason': reason,
            'entry_choppy': self.entry_choppy,
            'entry_momentum': self.entry_momentum,
            'exit_choppy': choppy
        })

        self.position = None
        self.entry_price = 0

    def backtest(self, df):
        """Run backtest on the dataframe"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Split into previous and current day
        prev_day_df = df[df['timestamp'].dt.date == pd.to_datetime(self.datefrom).date()]
        current_day_df = df[df['timestamp'].dt.date == pd.to_datetime(self.dateto).date()]

        # Generate Renko bricks for both days
        all_bricks = self.generate_renko(pd.concat([prev_day_df, current_day_df]))
        prev_bricks = all_bricks[all_bricks['timestamp'].dt.date == pd.to_datetime(self.datefrom).date()].reset_index(
            drop=True)
        current_bricks = all_bricks[all_bricks['timestamp'].dt.date == pd.to_datetime(self.dateto).date()].reset_index(
            drop=True)

        print("Previous day bricks:", end ='\n')
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Don't limit the display width
        pd.set_option('display.max_colwidth', None)
        print(prev_bricks)
        # Store current day bricks for display
        self.current_day_bricks = current_bricks.copy()

        # Initialize trading variables
        self.position = None
        self.entry_price = 0
        self.entry_time = None
        chart_pattern = []

        # Warm up indicators with previous day data
        for i in range(len(prev_bricks)):
            window = prev_bricks.iloc[:i + 1]
            choppy, momentum = self.calculate_indicators(window)
            self.indicators_history.append({
                'timestamp': prev_bricks.iloc[i]['timestamp'],
                'choppy': choppy,
                'momentum': momentum
            })

        # Run trading logic on current day
        for idx, brick in current_bricks.iterrows():
            ts = brick['timestamp']
            direction = brick['direction']
            price = brick['close']

            # Calculate indicators using both days' bricks (for ALL timestamps)
            window = pd.concat([prev_bricks, current_bricks.iloc[:idx + 1]])
            choppy, momentum = self.calculate_indicators(window)

            # Store indicators for display (including 9:15 AM)
            self.indicators_history.append({
                'timestamp': ts,
                'choppy': choppy,
                'momentum': momentum
            })

            # Skip trade decisions during 9:15-9:16 (but keep indicators)
            if time(9, 15) <= ts.time() < time(9, 16):
                continue

            # Update chart pattern (max 2 elements)
            chart_pattern = (chart_pattern + [direction])[-2:]

            # Buy logic (only after 9:16)
            if self.position is None and len(chart_pattern) == 2:
                if chart_pattern in [[1, 1], [0, 1]]:  # Pattern match
                    if not self.choppy_low <= choppy <= self.choppy_high:  # Choppy filter
                        if momentum > 0:  # Momentum filter
                            self.open_position(ts, price, choppy, momentum)

            # Sell logic
            elif self.position == 'LONG':
                if len(chart_pattern) >= 1 and chart_pattern[-1] == 0:  # Pattern exit
                    self.close_position(ts, price, 'PATTERN_EXIT', choppy)
                elif self.choppy_low <= choppy <= self.choppy_high:  # Choppy exit
                    self.close_position(ts, price, 'CHOPPY_EXIT', choppy)

        # Square off any remaining position at EOD
        if self.position == 'LONG' and not current_bricks.empty:
            last_price = current_bricks.iloc[-1]['close']
            choppy, _ = self.calculate_indicators(pd.concat([prev_bricks, current_bricks]))
            self.close_position(current_bricks.iloc[-1]['timestamp'], last_price, 'EOD', choppy)

        # Return results (empty DataFrame if no trades)
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame(columns=[
            'symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'qty', 'gross_pnl', 'charges', 'net_pnl', 'reason',
            'entry_choppy', 'entry_momentum', 'exit_choppy'
        ])

if __name__ == "__main__":
    data = pd.read_csv("bufferFiles/backtestdata.csv")
    backtester = RenkoBacktester(
        symbol="NSE:JMFINANCIL-EQ",
        brick_size=0.25,
        qty=18,
        prev_day_renko_close=142.00,
        datefrom = '2025-06-19',
        dateto =  '2025-06-20'
    )

    # Run backtest
    results = backtester.backtest(data)

    # Print results with error handling
    if not results.empty:
        print("\nTrade Results:")
        print(results[['entry_time', 'exit_time', 'entry_price', 'exit_price',
                       'entry_choppy', 'entry_momentum', 'exit_choppy',
                       'net_pnl', 'reason']])

        total_net = results['net_pnl'].sum()
        print(f"\nTotal Net P&L: {total_net:.2f}")
        print(f"Number of Trades: {len(results)}")
        print(f"Average P&L per Trade: {total_net / len(results) if len(results) > 0 else 0:.2f}")
    else:
        print("No trades were executed during the backtest period.")

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Don't limit the display width
    pd.set_option('display.max_colwidth', None)
    # Print all computed indicators (for current day only, with index starting at 0)
    print("\nAll Computed Indicators (Current Day):")
    indicators_df = pd.DataFrame(backtester.indicators_history)
    current_day_mask = indicators_df['timestamp'].dt.date == pd.to_datetime(backtester.dateto).date()
    current_day_indicators = indicators_df[current_day_mask].reset_index(drop=True)
    print(current_day_indicators)

    # Print current day's Renko bricks
    print("\nCurrent Day Renko Bricks:")
    print(backtester.current_day_bricks)


