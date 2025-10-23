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

        # Trading parameters
        self.choppy_period = 14
        self.choppy_period_5 = 5
        self.choppy_period_3 = 3
        self.momentum_period = 7
        self.choppy_low = 40
        self.choppy_high = 100
        self.choppy_5_threshold = 40
        self.choppy_3_threshold = 40

        # Golden ratio filter parameters
        self.golden_ratio = 1.618
        self.golden_ratio_period = 14

        # RSI parameters
        self.rsi_period = 7
        self.rsi_buy_threshold = 50

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

    def calculate_rsi_traditional(self, closes):
        """
        Calculate RSI using traditional method with proper EMA
        This is a standalone function that calculates RSI for a series of closes
        """
        if len(closes) < self.rsi_period + 1:
            return [50] * len(closes)

        # Calculate price changes
        price_changes = np.diff(closes)

        # Separate gains and losses
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)

        rsi_values = []

        # Calculate initial averages using SMA for first RSI_PERIOD values
        initial_avg_gain = np.mean(gains[:self.rsi_period])
        initial_avg_loss = np.mean(losses[:self.rsi_period])

        # Calculate first RSI
        if initial_avg_loss == 0:
            first_rsi = 100
        else:
            rs = initial_avg_gain / initial_avg_loss
            first_rsi = 100 - (100 / (1 + rs))

        rsi_values.append(first_rsi)

        # Calculate subsequent RSI values using EMA
        avg_gain = initial_avg_gain
        avg_loss = initial_avg_loss

        for i in range(self.rsi_period, len(gains)):
            # EMA calculation
            alpha = 1.0 / self.rsi_period
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

        return rsi_values

    def calculate_rsi(self, brick_index):
        """
        Calculate RSI using the traditional method
        Returns RSI value for the current brick
        """
        if brick_index < self.rsi_period:
            return 50  # Return neutral RSI if not enough data

        # Get enough historical data to calculate RSI properly
        start_idx = max(0, brick_index - 50)  # Get more history for better EMA calculation
        end_idx = brick_index + 1

        recent_bricks = self.all_bricks_continuous.iloc[start_idx:end_idx]

        if len(recent_bricks) < self.rsi_period + 1:
            return 50

        closes = recent_bricks['close'].values
        rsi_values = self.calculate_rsi_traditional(closes)

        # Return the last RSI value
        return rsi_values[-1] if rsi_values else 50

    def calculate_golden_ratio_filter(self, brick_index):
        """
        Calculate the golden ratio filter using the last 14 bricks
        Returns True if the ratio of green:red bricks >= golden ratio, False otherwise
        """
        if brick_index < self.golden_ratio_period:
            return False  # Not enough data, don't allow trading

        # Get the last 14 bricks (including current)
        start_idx = brick_index - self.golden_ratio_period + 1
        end_idx = brick_index + 1

        recent_bricks = self.all_bricks_continuous.iloc[start_idx:end_idx]

        if len(recent_bricks) < self.golden_ratio_period:
            return False

        # Count green (direction=1) and red (direction=0) bricks
        green_count = (recent_bricks['direction'] == 1).sum()
        red_count = (recent_bricks['direction'] == 0).sum()

        # Calculate ratio (green:red)
        if red_count == 0:
            # All green bricks - this exceeds golden ratio
            green_red_ratio = float('inf')
            ratio_passes = True
        else:
            green_red_ratio = green_count / red_count
            ratio_passes = green_red_ratio >= self.golden_ratio

        return [ratio_passes, green_red_ratio, green_count, red_count]

    def calculate_indicators_continuous(self, brick_index):
        """Calculate choppiness, momentum, and RSI using continuous brick history"""
        if brick_index < max(self.choppy_period, self.momentum_period, self.rsi_period, self.choppy_period_5, self.choppy_period_3):
            return 0, 0, 0, 1, 50

        # Calculate choppy_14
        start_idx = max(0, brick_index - self.choppy_period)
        end_idx = brick_index + 1
        recent = self.all_bricks_continuous.iloc[start_idx:end_idx]

        if len(recent) < self.choppy_period:
            return 0, 0, 0, 1, 50

        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values

        if len(closes) <= 1:
            return 0, 0, 0, 1, 50

        tr = np.maximum.reduce([
            highs[1:] - lows[1:],
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        ])

        sum_tr = np.sum(tr)
        max_high = np.max(highs[1:])
        min_low = np.min(lows[1:])
        range_ = max_high - min_low

        choppy_14 = 100 * np.log10(sum_tr / range_) / np.log10(self.choppy_period) if range_ > 0 else 0

        # Calculate choppy_5
        start_idx_5 = max(0, brick_index - self.choppy_period_5)
        end_idx_5 = brick_index + 1
        recent_5 = self.all_bricks_continuous.iloc[start_idx_5:end_idx_5]

        if len(recent_5) < self.choppy_period_5:
            choppy_5 = 0
        else:
            highs_5 = recent_5['high'].values
            lows_5 = recent_5['low'].values
            closes_5 = recent_5['close'].values

            if len(closes_5) <= 1:
                choppy_5 = 0
            else:
                tr_5 = np.maximum.reduce([
                    highs_5[1:] - lows_5[1:],
                    np.abs(highs_5[1:] - closes_5[:-1]),
                    np.abs(lows_5[1:] - closes_5[:-1])
                ])

                sum_tr_5 = np.sum(tr_5)
                max_high_5 = np.max(highs_5)
                min_low_5 = np.min(lows_5)
                range_5 = max_high_5 - min_low_5

                choppy_5 = 100 * np.log10(sum_tr_5 / range_5) / np.log10(self.choppy_period_5) if range_5 > 0 else 0

        # Calculate choppy_3 (NEW)
        start_idx_3 = max(0, brick_index - self.choppy_period_3)
        end_idx_3 = brick_index + 1
        recent_3 = self.all_bricks_continuous.iloc[start_idx_3:end_idx_3]

        if len(recent_3) < self.choppy_period_3:
            choppy_3 = 0
        else:
            highs_3 = recent_3['high'].values
            lows_3 = recent_3['low'].values
            closes_3 = recent_3['close'].values

            if len(closes_3) <= 1:
                choppy_3 = 0
            else:
                tr_3 = np.maximum.reduce([
                    highs_3[1:] - lows_3[1:],
                    np.abs(highs_3[1:] - closes_3[:-1]),
                    np.abs(lows_3[1:] - closes_3[:-1])
                ])

                sum_tr_3 = np.sum(tr_3)
                max_high_3 = np.max(highs_3)
                min_low_3 = np.min(lows_3)
                range_3 = max_high_3 - min_low_3

                choppy_3 = 100 * np.log10(sum_tr_3 / range_3) / np.log10(self.choppy_period_3) if range_3 > 0 else 0

        # Momentum calculation (unchanged)
        momentum_start_idx = max(0, brick_index - self.momentum_period)
        if momentum_start_idx < len(self.all_bricks_continuous):
            momentum = closes[-1] - self.all_bricks_continuous.iloc[momentum_start_idx]['close']
        else:
            momentum = 1

        # RSI calculation (unchanged)
        rsi = self.calculate_rsi(brick_index)

        return choppy_14, choppy_5, choppy_3, momentum, rsi

    def open_position(self, ts, theoretical_price, choppy_14, choppy_5, choppy_3, momentum, rsi, golden_ratio_data):
        """Record new long position with slippage applied"""
        actual_entry_price = self.apply_slippage(theoretical_price, is_buy=True)

        self.position = 'LONG'
        self.entry_price = actual_entry_price
        self.theoretical_entry_price = theoretical_price
        self.entry_time = ts
        self.entry_choppy = choppy_14
        self.entry_choppy_5 = choppy_5
        self.entry_choppy_3 = choppy_3  # Add this line
        self.entry_momentum = momentum
        self.entry_rsi = rsi
        self.entry_golden_ratio_data = golden_ratio_data

    def close_position(self, ts, theoretical_price, reason, choppy_14, choppy_5, choppy_3, rsi):
        """Close existing position with slippage applied"""
        if self.position != 'LONG':
            return

        actual_exit_price = self.apply_slippage(theoretical_price, is_buy=False)
        exit_time = ts

        # Calculate P&L using actual executed prices (with slippage)
        gross_pnl = (actual_exit_price - self.entry_price) * self.qty
        charges = self.calculate_charges(self.entry_price * self.qty, actual_exit_price * self.qty)
        net_pnl = gross_pnl - charges

        # FIXED: Calculate slippage impact correctly
        # Both entry and exit slippage should represent COSTS (negative impact on P&L)

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
            'entry_choppy': self.entry_choppy,
            'entry_choppy_5': self.entry_choppy_5,
            'entry_choppy_3': self.entry_choppy_3,  # Add this line
            'entry_momentum': self.entry_momentum,
            'entry_rsi': self.entry_rsi,
            'exit_choppy': choppy_14,
            'exit_choppy_5': choppy_5,
            'exit_choppy_3': choppy_3,  # Add this line
            'exit_rsi': rsi,
            'entry_golden_ratio': self.entry_golden_ratio_data['ratio'],
            'entry_green_count': self.entry_golden_ratio_data['green_count'],
            'entry_red_count': self.entry_golden_ratio_data['red_count']
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
        """Run backtest with CONTINUOUS indicator calculations across all days"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        print("Generating continuous Renko bricks across all days...")
        self.all_bricks_continuous = self.generate_renko(df)
        print(f"Total bricks generated: {len(self.all_bricks_continuous)}")
        print(f"Slippage factor: {self.slippage} points per trade leg")
        print(f"Golden ratio filter: {self.golden_ratio} (period: {self.golden_ratio_period})")
        print(f"RSI period: {self.rsi_period}, RSI buy threshold: {self.rsi_buy_threshold}")
        print(f"Choppy 14 period: {self.choppy_period}, Choppy 5 period: {self.choppy_period_5}, Choppy 3 period: {self.choppy_period_3}")
        print(f"Choppy 5 threshold: {self.choppy_5_threshold}, Choppy 3 threshold: {self.choppy_3_threshold}")

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

                choppy_14, choppy_5, choppy_3, momentum, rsi = self.calculate_indicators_continuous(continuous_idx)

                # Calculate golden ratio filter
                golden_ratio_result = self.calculate_golden_ratio_filter(continuous_idx)

                if isinstance(golden_ratio_result, list) and len(golden_ratio_result) == 4:
                    golden_ratio_passes, golden_ratio_value, green_count, red_count = golden_ratio_result
                else:
                    golden_ratio_passes = golden_ratio_result
                    golden_ratio_value, green_count, red_count = 0, 0, 0

                golden_ratio_data = {
                    'passes': golden_ratio_passes,
                    'ratio': golden_ratio_value,
                    'green_count': green_count,
                    'red_count': red_count
                }

                self.indicators_history.append({
                    'timestamp': ts,
                    'choppy_14': choppy_14,
                    'choppy_5': choppy_5,
                    'choppy_3': choppy_3,
                    'momentum': momentum,
                    'rsi': rsi,
                    'golden_ratio_passes': golden_ratio_passes,
                    'golden_ratio_value': golden_ratio_value,
                    'green_count': green_count,
                    'red_count': red_count,
                    'continuous_index': continuous_idx,
                    'local_index': local_idx
                })

                # Mandatory square off at 3:15 PM
                if ts.time() >= self.square_off_time and self.position == 'LONG':
                    self.close_position(ts, theoretical_price, 'EOD', choppy_14, choppy_5, choppy_3, rsi)
                    print(f"MANDATORY square off at 3:15 PM: {ts} - Theoretical Price: {theoretical_price}")
                    continue

                if ts.time() >= self.square_off_time:
                    continue

                if time(9, 15) <= ts.time() < time(9, 16):
                    continue

                if is_last_brick_of_day:
                    print(f"SKIPPING trade decision on LAST BRICK of day at {ts}")
                    continue

                chart_pattern = (chart_pattern + [direction])[-2:]

                # Buy logic with golden ratio filter, RSI filter, and choppy_5 filter
                if self.position is None and len(chart_pattern) == 2:
                    if chart_pattern in [[1, 1], [0, 1]]:
                        if not self.choppy_low <= choppy_14 <= self.choppy_high:
                            if momentum > 0:
                                # Check choppy_5 condition
                                if choppy_5 < self.choppy_5_threshold:
                                    # NEW: Check choppy_3 condition
                                    if choppy_3 < self.choppy_3_threshold:
                                        # Check golden ratio filter
                                        if golden_ratio_passes:
                                            # Check RSI filter
                                            if rsi > self.rsi_buy_threshold:
                                                actual_entry_price = self.apply_slippage(theoretical_price, is_buy=True)
                                                print(f"Opening LONG position at {ts} - "
                                                      f"Theoretical Price: {theoretical_price}, Actual Price: {actual_entry_price:.2f} "
                                                      f"(Slippage: +{self.slippage}) - Golden Ratio: {golden_ratio_value:.3f} "
                                                      f"(G:{green_count}, R:{red_count}) - RSI: {rsi:.2f} - "
                                                      f"Choppy14: {choppy_14:.2f}, Choppy5: {choppy_5:.2f}, Choppy3: {choppy_3:.2f}")
                                                self.open_position(ts, theoretical_price, choppy_14, choppy_5, choppy_3,
                                                                   momentum,
                                                                   rsi, golden_ratio_data)
                                            else:
                                                print(f"SKIPPING entry at {ts} due to RSI filter - "
                                                      f"RSI: {rsi:.2f} <= {self.rsi_buy_threshold}")
                                        else:
                                            print(f"SKIPPING entry at {ts} due to Golden Ratio filter - "
                                                  f"Ratio: {golden_ratio_value:.3f} < {self.golden_ratio} "
                                                  f"(G:{green_count}, R:{red_count})")
                                    else:
                                        print(f"SKIPPING entry at {ts} due to Choppy3 filter - "
                                              f"Choppy3: {choppy_3:.2f} >= {self.choppy_3_threshold}")
                                else:
                                    print(f"SKIPPING entry at {ts} due to Choppy5 filter - "
                                          f"Choppy5: {choppy_5:.2f} >= {self.choppy_5_threshold}")

                # Sell logic
                elif self.position == 'LONG':
                    if len(chart_pattern) >= 1 and chart_pattern[-1] == 0:
                        actual_exit_price = self.apply_slippage(theoretical_price, is_buy=False)
                        print(f"Closing position due to PATTERN_EXIT at {ts} - "
                              f"Theoretical Price: {theoretical_price}, Actual Price: {actual_exit_price:.2f} "
                              f"(Slippage: -{self.slippage}) - RSI: {rsi:.2f}")
                        self.close_position(ts, theoretical_price, 'PATTERN_EXIT', choppy_14, choppy_5, choppy_3, rsi)
                    elif self.choppy_low <= choppy_14 <= self.choppy_high:
                        actual_exit_price = self.apply_slippage(theoretical_price, is_buy=False)
                        print(f"Closing position due to CHOPPY_EXIT at {ts} - "
                              f"Theoretical Price: {theoretical_price}, Actual Price: {actual_exit_price:.2f} "
                              f"(Slippage: -{self.slippage}) - RSI: {rsi:.2f}")
                        self.close_position(ts, theoretical_price, 'CHOPPY_EXIT', choppy_14, choppy_5, choppy_3, rsi)

            # Ensure all positions are squared off by 3:15 PM
            if self.position == 'LONG':
                try:
                    square_off_price, square_off_time = self.get_closest_price_at_time(current_day_df,
                                                                                       self.square_off_time)
                    last_continuous_idx = current_day_continuous_indices[-1] if current_day_continuous_indices else 0
                    choppy_14, choppy_5, choppy_3, _, rsi = self.calculate_indicators_continuous(last_continuous_idx)
                    self.close_position(square_off_time, square_off_price, 'EOD', choppy_14, choppy_5, choppy_3, rsi)
                    print(f"END OF DAY square off executed at {square_off_time}")
                except Exception as e:
                    last_price = current_day_df.iloc[-1]['close']
                    last_time = current_day_df.iloc[-1]['timestamp']
                    last_continuous_idx = current_day_continuous_indices[-1] if current_day_continuous_indices else 0
                    choppy_14, choppy_5, choppy_3, _, rsi = self.calculate_indicators_continuous(last_continuous_idx)
                    self.close_position(last_time, last_price, 'EMERGENCY_SQUARE_OFF', choppy_14, choppy_5, choppy_3,
                                        rsi)
                    print(f"EMERGENCY square off executed at {last_time}")
                    print(f"ERROR during square off: {e}")

        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame(columns=[
            'symbol',
            'entry_time',
            'exit_time',
            'entry_price',
            'exit_price',
            'theoretical_entry_price',
            'theoretical_exit_price',
            'qty',
            'gross_pnl',
            'charges',
            'net_pnl',
            'slippage_impact',
            'entry_slippage',
            'exit_slippage',
            'reason',
            'entry_choppy',
            'entry_choppy_5',
            'entry_choppy_3',
            'entry_momentum',
            'entry_rsi',
            'exit_choppy',
            'exit_choppy_3',
            'exit_rsi',
            'entry_golden_ratio',
            'entry_green_count',
            'entry_red_count'
        ])


if __name__ == "__main__":
    data = pd.read_csv("bufferFiles/backtestdata.csv")
    backtester = RenkoBacktester(
        symbol="NSE:JMFINANCIL-EQ",
        brick_size=0.5,
        qty=18,
        prev_day_renko_close=100.50,
        datefrom='2025-05-02',
        dateto='2025-06-24',
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
                       'entry_choppy', 'entry_momentum', 'entry_rsi', 'exit_choppy',
                        'net_pnl', 'reason']])

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

        # Golden ratio statistics
        if 'entry_golden_ratio' in results.columns:
            avg_golden_ratio = results['entry_golden_ratio'].mean()
            min_golden_ratio = results['entry_golden_ratio'].min()
            max_golden_ratio = results['entry_golden_ratio'].max()
            print(f"Average Golden Ratio at Entry: {avg_golden_ratio:.3f}")
            print(f"Min Golden Ratio at Entry: {min_golden_ratio:.3f}")
            print(f"Max Golden Ratio at Entry: {max_golden_ratio:.3f}")

        # RSI statistics
        if 'entry_rsi' in results.columns:
            avg_entry_rsi = results['entry_rsi'].mean()
            min_entry_rsi = results['entry_rsi'].min()
            max_entry_rsi = results['entry_rsi'].max()
            avg_exit_rsi = results['exit_rsi'].mean()
            print(f"Average Entry RSI: {avg_entry_rsi:.2f}")
            print(f"Min Entry RSI: {min_entry_rsi:.2f}")
            print(f"Max Entry RSI: {max_entry_rsi:.2f}")
            print(f"Average Exit RSI: {avg_exit_rsi:.2f}")

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

                # Calculate some performance metrics
                winning_trades = len(results[results['net_pnl'] > 0])
                losing_trades = len(results[results['net_pnl'] < 0])
                win_rate = (winning_trades / len(results)) * 100 if len(results) > 0 else 0

                if winning_trades > 0:
                    avg_win = results[results['net_pnl'] > 0]['net_pnl'].mean()
                else:
                    avg_win = 0

                if losing_trades > 0:
                    avg_loss = results[results['net_pnl'] < 0]['net_pnl'].mean()
                else:
                    avg_loss = 0

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

            # Save indicators history
            if backtester.indicators_history:
                indicators_df = pd.DataFrame(backtester.indicators_history)
                indicators_df.to_csv('bufferFiles/indicators_history.csv', index=False)
                print(f"Indicators history saved to 'bufferFiles/indicators_history.csv'")

        else:
            print("No trades were generated during the backtest period.")
            print("This could be due to:")
            print("- Market conditions not meeting entry criteria")
            print("- Golden ratio filter being too restrictive")
            print("- RSI filter being too restrictive")
            print("- Choppiness conditions not being met")
            print("- Insufficient price movement to generate required brick patterns")