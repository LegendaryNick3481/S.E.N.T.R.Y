"""
Architecture:
MasterBot() creates independent instances of StockBot() using coroutines.Each StockBot() takes its own set of independent
parameters (brickSize, timeframe, quantity ) & handles everything (buy, sell, candle conversion, choppiness calculation
database handling) etc etc related to the security assigned to it. Ticker data is sent to each StockBot() via the MasterBot()
"""
import asyncio
import pickle
import math
import numpy as np
import pandas as pd
import credentials as crs
import aiomysql as connector

from itertools import islice
from collections import deque

from fyers_apiv3 import fyersModel
from datetime import datetime, time
from decimal import Decimal, getcontext
from fyers_apiv3.FyersWebsocket import data_ws


class StockBot():
    activeTrades = 0
    maxActiveTrades = 6
    tradeLock = asyncio.Lock()
    getcontext().prec = 10

    def __init__(self, symbol_key, s, dbPool, ofyers_instance):
        self.registeredPnl = 0
        self.maxLoss = -100
        self.running = True

        # Multiple choppiness periods
        self.choppyPeriod3 = 3
        self.choppyPeriod7 = 7
        self.choppyPeriod14 = 14

        # Choppiness threshold
        self.choppyThreshold = 40

        self.buy = False
        self.dbPool = dbPool
        self.symbol_key = symbol_key  # The full symbol key like "NSE:BANKBARODA-EQ"
        self.ticker = symbol_key[4:]
        self.symbol = s['symbol'] if 'symbol' in s else symbol_key
        self.timeframe = s['timeframe']
        self.brickSize = s['brickSize']
        self.qty = s['quantity']
        self.initPrice = Decimal('0')
        self.buffer = list()
        self.chart = list()

        # Multiple choppiness arrays and values
        self.choppyArray3 = list()
        self.choppyArray7 = list()
        self.choppyArray14 = list()
        self.choppinessValue3 = Decimal('0')
        self.choppinessValue7 = Decimal('0')
        self.choppinessValue14 = Decimal('0')
        self.choppyInitialized3 = False
        self.choppyInitialized7 = False
        self.choppyInitialized14 = False
        self.square_off_done = False

        self.tradeLog = list()
        self.ofyers = ofyers_instance
        self.buyPrice = Decimal('0')
        self.sellPrice = Decimal('0')
        self.buyTime = ""
        self.sellTime = ""
        self.current_ltp = Decimal('0')
        self.candles = deque()
        self.bricks = deque(maxlen=100)
        self.bufferDF = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'direction'])
        self.messageBuffer = asyncio.Queue()
        self.candleBuffer = dict()
        self.prevDayLastDirection = None
        self.dataLock = asyncio.Lock()
        self.uploadDataFromFile()

    def uploadDataFromFile(self):
        try:
            with open(f'bufferFiles/{self.ticker}-restoreChartValues.pkl', "rb") as f:
                df = pickle.load(f)
                self.chart.extend(df['direction'].tolist())
                print(f"[{self.ticker}] Chart values initialised as: {self.chart}")
                brick_records = df.to_dict(orient='records')
                self.bricks = deque(brick_records, maxlen=100)

                if not df.empty:
                    """
                    executed if we restart the script
                    """
                    self.initPrice = Decimal(str(df.iloc[-1]['close']))
                    self.prevDayLastDirection = df.iloc[-1]['direction']
                    print(
                        f"[{self.ticker}] Current Day's Brick Close Price [Taken For Program Restart]: {self.initPrice}")
                    print(
                        f"[{self.ticker}] Current Day's Brick Close Color [Taken For Program Restart]: {self.prevDayLastDirection}")
                    print(f'[{self.ticker}] Chart data points restored successfully')
                else:
                    """
                    executed on day start 
                    """
                    with open(f'bufferFiles/{self.ticker}-getDayOpening.pkl', "rb") as f:
                        frame = pickle.load(f)
                        self.initPrice = Decimal(str(frame.iloc[-1]['close']))
                        self.prevDayLastDirection = frame.iloc[-1]['direction']
                        print(f"[{self.ticker}] Previous Day's Brick Close Price: {self.initPrice}")
                        print(f"[{self.ticker}] Previous Day's Brick Close Color: {self.prevDayLastDirection}")

        except FileNotFoundError:
            print(f'[{self.ticker}] restoreChartValues.pkl does not exist')

        try:
            with open(f"bufferFiles/{self.ticker}-restoreBricksForCI.pkl", "rb") as f:
                self.bufferDF = pickle.load(f)
                buffer_records = self.bufferDF.to_dict(orient='records')
                self.buffer = deque(buffer_records, maxlen=100)
                print(f"[{self.ticker}] The renko dataFrame is initialized with previous day bricks")
        except FileNotFoundError:
            print(f"[{self.ticker}] No renko bricks from yesterday found.")

    def handle_message(self, message):
        f = asyncio.run_coroutine_threadsafe(
            self.messageBuffer.put(message),
            self.loop
        )

    def calculateIntradayCharges(self, buy_value, sell_value, brokerage_rate=0.03, max_brokerage=20):
        turnover = buy_value + sell_value
        brokerage_buy = min(max_brokerage, (brokerage_rate / 100) * buy_value)
        brokerage_sell = min(max_brokerage, (brokerage_rate / 100) * sell_value)
        total_brokerage = brokerage_buy + brokerage_sell
        etc = turnover * (325 / 1e7)
        stt = sell_value * 0.00025
        sebi_charges = turnover * (10 / 1e7)
        stamp_duty = buy_value * 0.00003
        gst = 0.18 * (total_brokerage + etc)
        nse_ipft = turnover * (10 / 1e7)
        return round(total_brokerage + etc + stt + sebi_charges + stamp_duty + gst + nse_ipft, 2)

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.tasks = [
            asyncio.create_task(self.convToCandles()),
            asyncio.create_task(self.plotChart()),
            asyncio.create_task(self.indicatorLoop()),
            asyncio.create_task(self.buyLogic()),
            asyncio.create_task(self.sellLogic()),
            asyncio.create_task(self.storeToDatabase())
        ]
        print(f"[{self.symbol}] All bot tasks started")

    async def convToCandles(self):
        current_candle_time = None
        current_candle_data = None

        while self.running:
            message = await self.messageBuffer.get()
            ltp = message.get('ltp')
            if not ltp:
                continue

            ltp_decimal = Decimal(str(ltp))
            self.current_ltp = ltp_decimal

            ts = message.get('last_traded_time') or message.get('exch_feed_time')
            if isinstance(ts, str):
                ts = float(ts)
            dt = datetime.fromtimestamp(ts)

            minute_bucket = (dt.minute // self.timeframe) * self.timeframe
            candle_time = dt.replace(minute=minute_bucket, second=0, microsecond=0)

            async with self.dataLock:
                if candle_time != current_candle_time:
                    if current_candle_data is not None:
                        self.candles.append({
                            'timestamp': current_candle_time,
                            'open': str(current_candle_data['open']),
                            'high': str(current_candle_data['high']),
                            'low': str(current_candle_data['low']),
                            'close': str(current_candle_data['close']),
                        })

                    current_candle_time = candle_time
                    current_candle_data = {
                        'open': ltp_decimal,
                        'high': ltp_decimal,
                        'low': ltp_decimal,
                        'close': ltp_decimal,
                    }

                else:
                    if ltp_decimal > current_candle_data['high']:
                        current_candle_data['high'] = ltp_decimal
                    if ltp_decimal < current_candle_data['low']:
                        current_candle_data['low'] = ltp_decimal
                    current_candle_data['close'] = ltp_decimal

    async def plotChart(self):
        lastProcessedIdx = 0
        lastDirection = self.prevDayLastDirection
        lastRenkoClose = None

        while self.running:
            await asyncio.sleep(0.1)

            async with self.dataLock:
                if len(self.candles) <= lastProcessedIdx:
                    continue
                current_candles = [self.candles[i] for i in range(lastProcessedIdx, len(self.candles))]
                lastProcessedIdx = len(self.candles)

            new_bricks = []
            chart_directions = []

            for candle in current_candles:
                ts = candle['timestamp']
                price = Decimal(str(candle['close']))

                if lastRenkoClose is None:
                    lastRenkoClose = Decimal(str(self.initPrice))

                while True:
                    diff = price - lastRenkoClose
                    brickSize = Decimal(str(self.brickSize))

                    if abs(diff) < brickSize:
                        break

                    if lastDirection is None:
                        direction = 1 if diff > 0 else 0
                        newClose = lastRenkoClose + (brickSize if direction else -brickSize)
                        renkoOpen = lastRenkoClose
                        renkoHigh = max(renkoOpen, newClose)
                        renkoLow = min(renkoOpen, newClose)
                        lastDirection = direction
                        lastRenkoClose = newClose

                    elif lastDirection == 1:
                        if diff >= brickSize:
                            direction = 1
                            newClose = lastRenkoClose + brickSize
                            renkoOpen = lastRenkoClose
                            renkoHigh = newClose
                            renkoLow = renkoOpen
                            lastRenkoClose = newClose

                        elif diff <= -2 * brickSize:
                            direction = 0
                            newClose = lastRenkoClose - 2 * brickSize
                            renkoOpen = lastRenkoClose - brickSize
                            renkoHigh = renkoOpen
                            renkoLow = newClose
                            lastRenkoClose = newClose
                            lastDirection = 0

                        else:
                            break

                    elif lastDirection == 0:
                        if diff <= -brickSize:
                            direction = 0
                            newClose = lastRenkoClose - brickSize
                            renkoOpen = lastRenkoClose
                            renkoLow = newClose
                            renkoHigh = renkoOpen
                            lastRenkoClose = newClose

                        elif diff >= 2 * brickSize:
                            direction = 1
                            newClose = lastRenkoClose + 2 * brickSize
                            renkoOpen = lastRenkoClose + brickSize
                            renkoLow = renkoOpen
                            renkoHigh = newClose
                            lastRenkoClose = newClose
                            lastDirection = 1

                        else:
                            break

                    newRow = {
                        'timestamp': ts,
                        'open': float(renkoOpen),
                        'high': float(renkoHigh),
                        'low': float(renkoLow),
                        'close': float(newClose),
                        'direction': direction
                    }

                    new_bricks.append(newRow)

                    if ts.time() >= time(9, 16) and ts.time() <= time(15, 31):
                        chart_directions.append(direction)

            # Append results in bulk, with minimal lock time
            async with self.dataLock:
                self.bricks.extend(new_bricks)
                if chart_directions:
                    self.chart.extend(chart_directions)
                    print(f"[{self.ticker}] B: {self.chart}")

    def calculate_choppiness(self, data, period):
        """
        Calculate choppiness index for given data and period
        """
        epsilon = 1e-10

        if len(data) < period + 1:
            return 50.0  # Default value when insufficient data

        # Take the most recent (period + 1) data points
        recent = data[-(period + 1):]

        highs = [x['high'] for x in recent]
        lows = [x['low'] for x in recent]
        closes = [x['close'] for x in recent]

        prev_closes = closes[:-1]
        curr_highs = highs[1:]
        curr_lows = lows[1:]

        curr_highs = np.array(curr_highs)
        curr_lows = np.array(curr_lows)
        prev_closes = np.array(prev_closes)

        tr = np.maximum.reduce([
            curr_highs - curr_lows,
            np.abs(curr_highs - prev_closes),
            np.abs(curr_lows - prev_closes)
        ])

        price_range = np.max(curr_highs) - np.min(curr_lows)

        if price_range <= 0 or np.isnan(price_range) or np.sum(tr) == 0:
            return 50.0
        else:
            atr_sum = np.sum(tr)
            ratio = atr_sum / (price_range + epsilon)
            chop = 100 * np.log10(ratio) / np.log10(period)
            return round(float(np.clip(chop, 0, 100)), 2)

    async def indicatorLoop(self):
        """
        Calculate multiple choppiness indicators (3, 7, 14 periods)
        """
        while self.running:
            await asyncio.sleep(0.1)

            async with self.dataLock:
                # Combine yesterday's buffer + today's new bricks
                combined_data = list(self.buffer) + list(self.bricks)

                # Calculate choppiness for each period
                if len(combined_data) >= self.choppyPeriod3 + 1:
                    new_chop3 = self.calculate_choppiness(combined_data, self.choppyPeriod3)
                    if not self.choppyArray3 or new_chop3 != self.choppinessValue3:
                        self.choppinessValue3 = new_chop3
                        self.choppyArray3.append(self.choppinessValue3)
                        print(f"[{self.ticker}] C3:", self.choppyArray3)
                    self.choppyInitialized3 = True

                if len(combined_data) >= self.choppyPeriod7 + 1:
                    new_chop7 = self.calculate_choppiness(combined_data, self.choppyPeriod7)
                    if not self.choppyArray7 or new_chop7 != self.choppinessValue7:
                        self.choppinessValue7 = new_chop7
                        self.choppyArray7.append(self.choppinessValue7)
                        print(f"[{self.ticker}] C7:", self.choppyArray7)
                    self.choppyInitialized7 = True

                if len(combined_data) >= self.choppyPeriod14 + 1:
                    new_chop14 = self.calculate_choppiness(combined_data, self.choppyPeriod14)
                    if not self.choppyArray14 or new_chop14 != self.choppinessValue14:
                        self.choppinessValue14 = new_chop14
                        self.choppyArray14.append(self.choppinessValue14)
                        print(f"[{self.ticker}] C14:", self.choppyArray14)
                    self.choppyInitialized14 = True

    async def buyLogic(self):
        while self.running:
            await asyncio.sleep(0.1)

            # Step 1: Get signal variables
            async with self.dataLock:
                # Get last 4 chart values for pattern matching
                currentChart = self.chart[-4:] if len(self.chart) >= 4 else []
                currentChoppy3 = self.choppinessValue3
                currentChoppy7 = self.choppinessValue7
                currentChoppy14 = self.choppinessValue14
                localBuy = self.buy
                choppyReady3 = self.choppyInitialized3
                choppyReady7 = self.choppyInitialized7
                choppyReady14 = self.choppyInitialized14

            # Step 2: Check buy conditions
            # Pattern: [0,1,1,1] or [1,1,1,1]
            valid_patterns = ([0, 1, 1, 1], [1, 1, 1, 1])
            pattern_match = currentChart in valid_patterns

            # All choppiness values should be < 40
            all_choppiness_low = (
                    (not choppyReady3 or currentChoppy3 < self.choppyThreshold) and
                    (not choppyReady7 or currentChoppy7 < self.choppyThreshold) and
                    (not choppyReady14 or currentChoppy14 < self.choppyThreshold)
            )

            isOpportunity = (
                    not localBuy
                    and pattern_match
                    and all_choppiness_low
            )

            if not isOpportunity:
                continue

            # Step 3: Try reserving a trade slot
            slot_reserved = False

            async with StockBot.tradeLock:
                if StockBot.activeTrades < StockBot.maxActiveTrades:
                    StockBot.activeTrades += 1
                    slot_reserved = True
                    print(f"Active Trades: {StockBot.activeTrades}")

            if not slot_reserved:
                continue

            try:
                async with self.dataLock:
                    buy_data = {
                        "symbol": self.symbol_key,
                        "qty": self.qty,
                        "type": 2,
                        "side": 1,
                        "productType": "INTRADAY",
                        "limitPrice": 0,
                        "stopPrice": 0,
                        "validity": "DAY",
                        "disclosedQty": 0,
                        "offlineOrder": False
                    }

                response = await self.ofyers.place_order(data=buy_data)
                print(f"[{self.symbol_key}] Buy order response:", response)

                if response.get('s') == 'ok':
                    async with self.dataLock:
                        self.buy = True
                        self.buyPrice = self.current_ltp
                        self.buyTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(
                            f"[{self.symbol_key}] BUY executed at {self.buyPrice} - Pattern: {currentChart}, C3:{currentChoppy3}, C7:{currentChoppy7}, C14:{currentChoppy14}")
                else:
                    print(f"[{self.symbol_key}] Buy failed.")
                    async with StockBot.tradeLock:
                        StockBot.activeTrades = max(0, StockBot.activeTrades - 1)

            except Exception as e:
                print(f"[{self.symbol_key}] Buy exception: {e}")

    async def sellLogic(self):
        while self.running:
            await asyncio.sleep(0.1)

            if self.square_off_done:
                continue

            async with self.dataLock:
                currentChart = self.chart[-1:] if len(self.chart) >= 1 else []
                currentChoppy3 = self.choppinessValue3
                currentChoppy7 = self.choppinessValue7
                currentChoppy14 = self.choppinessValue14
                localBuy = self.buy
                choppyReady3 = self.choppyInitialized3
                choppyReady7 = self.choppyInitialized7
                choppyReady14 = self.choppyInitialized14

            # Sell conditions: red brick [0] OR any choppiness > 40 OR end of day
            red_brick = currentChart == [0]

            any_choppiness_high = (
                    (choppyReady3 and currentChoppy3 > self.choppyThreshold) or
                    (choppyReady7 and currentChoppy7 > self.choppyThreshold) or
                    (choppyReady14 and currentChoppy14 > self.choppyThreshold)
            )

            end_of_day = datetime.now().time() > time(15, 5, 0)

            should_sell = (
                    localBuy and (
                    red_brick or
                    any_choppiness_high or
                    end_of_day
            )
            )

            if not should_sell:
                continue

            try:
                async with self.dataLock:
                    response = await self.ofyers.exit_positions(data={'id': self.symbol_key + '-INTRADAY'})

                print(f"[{self.symbol_key}] Sell order response:", response)

                if response.get('s') == 'ok':
                    async with self.dataLock:
                        self.buy = False
                        self.sellPrice = self.current_ltp
                        self.sellTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        sell_reason = ""
                        if red_brick:
                            sell_reason = "Red Brick"
                        elif any_choppiness_high:
                            sell_reason = f"High Choppiness (C3:{currentChoppy3}, C7:{currentChoppy7}, C14:{currentChoppy14})"
                        elif end_of_day:
                            sell_reason = "End of Day"

                        print(f"[{self.symbol_key}] SELL executed at {self.sellPrice} - Reason: {sell_reason}")

                        tradeData = {
                            'security': self.symbol_key,
                            'qty': self.qty,
                            'buyPrice': float(self.buyPrice),
                            'sellPrice': float(self.sellPrice),
                            'buytime': self.buyTime,
                            'selltime': self.sellTime,
                            'grossPnl': 0,
                            'regulatoryFees': 0,
                            'netPnl': 0
                        }
                        buyValue = tradeData['buyPrice'] * self.qty
                        sellValue = tradeData['sellPrice'] * self.qty
                        self.tradeLog.append(tradeData)
                        print(tradeData)

                    grossPnl = sellValue - buyValue
                    netPnl = grossPnl - self.calculateIntradayCharges(buyValue, sellValue)
                    self.registeredPnl += netPnl
                    if float(self.registeredPnl) <= float(self.maxLoss) or end_of_day:
                        self.square_off_done = True
                        await self.terminate()
                else:
                    print(f"[{self.symbol_key}] Sell failed.")

            except Exception as e:
                print(f"[{self.symbol_key}] Sell exception: {e}")

            finally:
                async with StockBot.tradeLock:
                    StockBot.activeTrades = max(0, StockBot.activeTrades - 1)
                    print(f"[{self.symbol_key}] Released sell slot. Active Trades: {StockBot.activeTrades}")

    async def storeToDatabase(self):
        closeTime = time(15, 31)
        eodBrickSaved = False

        if not self.dbPool:
            print(f"[{self.symbol_key}] Database pool not available")
            return

        try:
            async with self.dbPool.acquire() as conn:
                async with conn.cursor() as cursor:
                    while True:
                        now = datetime.now()

                        if self.tradeLog:
                            for t in self.tradeLog:
                                t['grossPnl'] = (t['sellPrice'] - t['buyPrice']) * t['qty']
                                t['regulatoryFees'] = self.calculateIntradayCharges(
                                    t['buyPrice'] * t['qty'],
                                    t['sellPrice'] * t['qty']
                                )
                                t['netPnl'] = t['grossPnl'] - t['regulatoryFees']

                                await cursor.execute('''
                                    INSERT INTO mark6 (
                                        security, qty, buyprice, sellprice, buytime, selltime,
                                        grossPnl, regulatoryFees, netPnl
                                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ''', (
                                    t['security'],
                                    int(t['qty']),
                                    float(t['buyPrice']),
                                    float(t['sellPrice']),
                                    str(t['buytime']),
                                    str(t['selltime']),
                                    float(t['grossPnl']),
                                    float(t['regulatoryFees']),
                                    float(t['netPnl'])
                                ))
                            self.tradeLog.clear()

                        elif now.time() >= closeTime and not eodBrickSaved:
                            # Save chart debug info
                            print(f'[{self.symbol_key}] DB write done, exiting storeToDatabase')
                            # Save day-end brick close
                            async with self.dataLock:
                                date_str = now.date().isoformat()
                                if self.bricks:
                                    dayBrickClose = self.bricks[-1]['close']
                                    await cursor.execute(
                                        '''UPDATE dayBrickClose SET date = %s , dayBrickClose =%s WHERE security = %s''',
                                        (date_str, dayBrickClose, self.symbol_key))
                                    print(f"[{self.symbol_key}] EOD Renko saved: {dayBrickClose}")
                                else:
                                    print(f"[{self.symbol_key}] renkoDF empty at EOD")

                            eodBrickSaved = True
                            return

                        await asyncio.sleep(1)

        except Exception as e:
            print(f"[{self.symbol_key}] Database error:", e)

    async def terminate(self):
        """
        Terminate the class instance if the bot has incurred loss greater than the max loss allowed
        """
        print(f'[{self.ticker}] Terminated Bot Action Reason')
        self.running = False


if __name__ == '__main__':
    pass