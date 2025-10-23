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
from datetime import datetime, time
from fyers_apiv3.FyersWebsocket import data_ws
from collections import deque
from fyers_apiv3 import fyersModel
from decimal import Decimal, getcontext

class StockBot():
    activeTrades = 0
    maxActiveTrades = 2 # Corresponds to approx 1000 rs in market
    tradeLock = asyncio.Lock()
    getcontext().prec = 10
    def __init__(self, symbol_key, s, dbPool, ofyers_instance):
        self.debugChart = list()
        self.debugBuyIndex = list()
        self.debugSellIndex = list()
        self.debugLists = dict()
        self.debugIndices = dict()
        self.choppyPeriod = 14
        self.momentumPeriod = 7
        self.choppyLow = 38
        self.choppyHigh = 100
        self.buy = False
        self.dbPool = dbPool
        self.symbol_key = symbol_key  # The full symbol key like "NSE:BANKBARODA-EQ"
        self.ticker = symbol_key[4:]
        self.symbol = s['symbol'] if 'symbol' in s else symbol_key
        self.timeframe = s['timeframe']
        self.brickSize = s['brickSize']
        self.prevDayClose = s['prevDayClose']
        self.qty = s['quantity']
        self.initPrice = 0
        self.chart = list()
        self.choppyArray = list()
        self.momentumArray = list()
        self.tradeLog = list()
        self.choppyInitialized = False
        self.momentumInitialized = False
        self.choppinessValue = 0 # Continue without choppineess
        self.momentumValue = 0
        self.ofyers = ofyers_instance
        self.buyPrice = 0
        self.sellPrice = 0
        self.buyTime = ""
        self.sellTime = ""
        self.current_ltp = 0
        self.candles = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
        self.renkoDF = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'direction'])
        self.bufferDF = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'direction'])
        self.messageBuffer = deque(maxlen=1000)
        self.candleBuffer = dict()
        self.dataLock = asyncio.Lock()
        self.uploadDataFromFile()

    def uploadDataFromFile(self):
        try:
            with open(f'bufferFiles/{self.ticker}-restoreChartValues.pkl', "rb") as f:
                df = pickle.load(f)
                self.chart.extend(df['direction'].tolist())
                print(f"[{self.ticker}] Chart values initialised as: {self.chart}")
                self.renkoDF = df.copy()

                if not df.empty:
                    """
                    executed if we restart the script
                    """
                    self.initPrice = Decimal(str(df.iloc[-1]['close']))
                    print(f"[{self.ticker}] Today's brick opening price : {self.initPrice}")
                    print(f'[{self.ticker}] Chart data points restored successfully')
                else:
                    """
                    executed on day start 
                    """
                    with open(f'bufferFiles/{self.ticker}-getDayOpening.pkl', "rb") as f:
                        frame = pickle.load(f)
                        self.initPrice = Decimal(str(frame.iloc[-1]['close']))
                        print(f"[{self.ticker}] Today's brick opening price : {self.initPrice}")

        except FileNotFoundError:
            print(f'[{self.ticker}] restoreChartValues.pkl does not exist')

        try:
            self.bufferDF = pd.read_csv(
                f"bufferFiles/{self.ticker}-restoreBricksForCI.csv").dropna(axis=1, how='all')
            print(f"[{self.ticker}] The renko dataFrame is initialized with previous day bricks")
        except FileNotFoundError:
            print(f"[{self.ticker}] No renko bricks from yesterday found.")

    def handle_message(self, message):
        self.messageBuffer.append(message)

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

    async def convToCandles(self):
        while True:
            await asyncio.sleep(0.05)
            async with self.dataLock:
                while self.messageBuffer:
                    try:
                        message = self.messageBuffer.popleft()
                        ltp_value = message.get('ltp')
                        if not ltp_value:
                            continue

                        self.current_ltp = ltp_value
                        ts = message.get('last_traded_time') or message.get('exch_feed_time')
                        if isinstance(ts, str):
                            ts = float(ts)
                        dt = datetime.fromtimestamp(ts)
                        minute_bucket = (dt.minute // self.timeframe) * self.timeframe
                        now = dt.replace(minute=minute_bucket, second=0, microsecond=0)

                        if now not in self.candleBuffer:
                            self.candleBuffer[now] = {'open': ltp_value, 'high': ltp_value, 'low': ltp_value,
                                                      'close': ltp_value}
                        else:
                            c = self.candleBuffer[now]
                            c['high'] = max(c['high'], ltp_value)
                            c['low'] = min(c['low'], ltp_value)
                            c['close'] = ltp_value

                        oldKeys = [ts for ts in self.candleBuffer if ts < now]
                        for oldTs in oldKeys:
                            c = self.candleBuffer.pop(oldTs)
                            newRow = {
                                'timestamp': oldTs,
                                'open': c['open'],
                                'high': c['high'],
                                'low': c['low'],
                                'close': c['close']
                            }
                            newDF = pd.DataFrame([newRow]).dropna(how='all', axis=1)
                            if not newDF.isna().all(axis=1).all():
                                self.candles = pd.concat([self.candles, newDF], ignore_index=True)
                    except Exception as e:
                        print(f"[{self.symbol_key}] Error processing message:", e)

    async def plotChart(self):

        lastProcessedIdx = 0
        lastDirection = None
        lastRenkoClose = None

        while True:
            await asyncio.sleep(0.05)

            async with self.dataLock:
                if len(self.candles) <= lastProcessedIdx:
                    continue
                current_candles = self.candles.iloc[lastProcessedIdx:].copy()
                lastProcessedIdx = len(self.candles)

            for _, row in current_candles.iterrows():
                ts = row['timestamp']
                price = Decimal(str(row['close']))

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

                    async with self.dataLock:
                        if ts.time() >= time(9, 16):
                            self.chart.append(direction)
                            print(f"[{self.ticker}] B: {self.chart}")
                        self.debugChart.append(direction)
                        self.renkoDF = pd.concat([self.renkoDF, pd.DataFrame([newRow])], ignore_index=True)

    async def indicatorLoop(self):
        buffer_processed = False
        epsilon = 1e-10  # avoid divide-by-zero

        while True:
            await asyncio.sleep(0.05)
            use_buffer = False

            async with self.dataLock:
                if not buffer_processed and len(self.bufferDF) >= self.choppyPeriod + 1:
                    recent = self.bufferDF.iloc[-(self.choppyPeriod + 1):]
                    use_buffer = True
                elif len(self.renkoDF) >= self.choppyPeriod + 1:
                    recent = self.renkoDF.iloc[-(self.choppyPeriod + 1):]
                else:
                    continue

            highs = recent['high'].astype(np.float64).values
            lows = recent['low'].astype(np.float64).values
            closes = recent['close'].astype(np.float64).values

            # ---- Choppiness Calculation ----
            prev_closes = closes[:-1]
            curr_highs = highs[1:]
            curr_lows = lows[1:]

            tr = np.maximum.reduce([
                curr_highs - curr_lows,
                np.abs(curr_highs - prev_closes),
                np.abs(curr_lows - prev_closes)
            ])

            price_range = np.max(curr_highs) - np.min(curr_lows)

            async with self.dataLock:
                if price_range <= 0 or np.isnan(price_range) or np.sum(tr) == 0:
                    self.choppinessValue = 50.0
                else:
                    atr_sum = np.sum(tr)
                    ratio = atr_sum / (price_range + epsilon)
                    chop = 100 * np.log10(ratio) / np.log10(self.choppyPeriod)
                    self.choppinessValue = round(float(np.clip(chop, 0, 100)), 2)

                    if not self.choppyArray or self.choppinessValue != self.choppyArray[-1]:
                        self.choppyArray.append(self.choppinessValue)
                        print(f"[{self.ticker}] C:", self.choppyArray)

                    self.choppyInitialized = True

                # ---- Momentum Calculation ----
                if len(closes) >= self.momentumPeriod + 1:
                    momentum_val = closes[-1] - closes[-(self.momentumPeriod + 1)]
                    self.momentumValue = round(float(momentum_val), 2)

                    if not self.momentumArray or self.momentumValue != self.momentumArray[-1]:
                        self.momentumArray.append(self.momentumValue)
                        print(f"[{self.ticker}] M:", self.momentumArray)

                    self.momentumInitialized = True
                else:
                    self.momentumValue = 0.0
                    self.momentumInitialized = False

                # ---- Clear buffer after use ----
                if use_buffer:
                    self.bufferDF = self.bufferDF.iloc[0:0]
                    buffer_processed = True

    async def buyLogic(self):
        while True:
            await asyncio.sleep(0.05)

            # Step 1: Get signal variables
            async with self.dataLock:
                currentIndex = len(self.chart) - 2
                currentChart = self.chart[-2:] if len(self.chart) >= 2 else []
                currentChoppy = self.choppinessValue
                currentMomentum = self.momentumValue
                localBuy = self.buy
                choppyReady = self.choppyInitialized
                momentumReady = self.momentumInitialized

            # Step 2: Check if there's a real buy opportunity
            isOpportunity = (not localBuy and currentChart in ([1, 1], [0, 1]) and (not choppyReady or not momentumReady or (not (self.choppyLow <= currentChoppy <= self.choppyHigh) and currentMomentum > 0)))

            if not isOpportunity:
                continue  # Skip this cycle, no opportunity

            # Step 3: Try reserving a trade slot
            async with StockBot.tradeLock:
                if StockBot.activeTrades >= StockBot.maxActiveTrades:
                    continue  # Someone else already filled the slot
                StockBot.activeTrades += 1  # Reserve slot
                print(f"Active Trades: {StockBot.activeTrades}")

            # Step 4: Place the order
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
                    self.debugBuyIndex.append(currentIndex)
                    self.buy = True
                    self.buyPrice = self.current_ltp
                    self.buyTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            except Exception as e:
                print(f"[{self.symbol_key}] Error placing buy order:", e)
                # Step 5: Rollback trade slot if failed
                async with StockBot.tradeLock:
                    StockBot.activeTrades -= 1

    async def sellLogic(self):
        while True:
            await asyncio.sleep(0.05)
            async with self.dataLock:
                currentIndex = len(self.chart) - 1
                currentChart = self.chart[-1:] if len(self.chart) >= 1 else []
                currentChoppy = self.choppinessValue
                localBuy = self.buy
                choppyReady = self.choppyInitialized

            if localBuy and (currentChart == [0] or (choppyReady and self.choppyLow <= currentChoppy <= self.choppyHigh) or datetime.now().time() > time(15, 5, 00)):
                async with self.dataLock:
                    try:
                        response = await self.ofyers.exit_positions(data={'id': self.symbol_key+'-INTRADAY'})
                        print(f"[{self.symbol_key}] Sell order response:", response)
                        self.debugSellIndex.append(currentIndex)
                        self.buy = False
                        self.sellPrice = self.current_ltp
                        self.sellTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        tradeData = {
                            'security': self.symbol_key,
                            'qty': self.qty,
                            'buyPrice': self.buyPrice,
                            'sellPrice': self.sellPrice,
                            'buytime': self.buyTime,
                            'selltime': self.sellTime,
                            'grossPnl': 0,
                            'regulatoryFees': 0,
                            'netPnl': 0
                        }
                        print(tradeData)
                        self.tradeLog.append(tradeData)

                        async with StockBot.tradeLock :
                            StockBot.activeTrades -= 1
                            print(f"Active Trades: {StockBot.activeTrades}")
                    except Exception as e:
                        print(f"[{self.symbol_key}] Error placing sell order:", e)

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
                                    t['security'], t['qty'], t['buyPrice'], t['sellPrice'],
                                    t['buytime'], t['selltime'],
                                    t['grossPnl'], t['regulatoryFees'], t['netPnl']
                                ))
                            self.tradeLog.clear()

                        elif now.time() >= closeTime and not eodBrickSaved:
                            # Save chart debug info
                            print(f'[{self.symbol_key}] DB write done, exiting storeToDatabase')

                            self.debugLists = {
                                'chart': self.chart,
                                'choppy': self.choppyArray
                            }
                            self.debugIndices = {
                                'buy': self.debugBuyIndex,
                                'sell': self.debugSellIndex
                            }
                            with open(f'bufferFiles/{self.ticker}-debugChart.txt', 'wb') as f:
                                pickle.dump(self.debugLists, f)
                            with open(f'bufferFiles/{self.ticker}-debugIndices.txt', 'wb') as f:
                                pickle.dump(self.debugIndices, f)

                            # Save day-end brick close
                            async with self.dataLock:
                                date_str = now.date().isoformat()
                                if not self.renkoDF.empty:
                                    dayBrickClose = self.renkoDF.iloc[-1]['close']
                                    await cursor.execute('''UPDATE dayBrickClose SET date = %s , dayBrickClose =%s WHERE security = %s''', (date_str,dayBrickClose,self.symbol_key))
                                    print(f"[{self.symbol_key}] EOD Renko saved: {dayBrickClose}")
                                else:
                                    print(f"[{self.symbol_key}] renkoDF empty at EOD")

                            eodBrickSaved = True
                            return

                        await asyncio.sleep(1)

        except Exception as e:
            print(f"[{self.symbol_key}] Database error:", e)

if __name__ == '__main__':
    pass
