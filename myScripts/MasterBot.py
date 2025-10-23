"""
Handles all the 'child' stockbot.StockBots(). Responsible for initializing & running all the stockbot.StockBots()
& also creating the database pool
"""
import VWAPStockBot as stockbot
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

class MasterBot:
    def __init__(self, symbolData):
        self.symbolData = symbolData
        self.symbols = list(symbolData.keys())
        self.pool = None
        self.password = 'tiger'
        self.dbName = 'nsetrades'
        self.bots = {}

        with open('bufferFiles/access_token.txt') as file:
            self.access_token = file.read().strip()

        self.ofyers = fyersModel.FyersModel(
            client_id=crs.client_id,
            token=self.access_token,
            is_async=True,
            log_path="bufferFiles/"
        )

        self.fyers = data_ws.FyersDataSocket(
            access_token=f'{crs.client_id}:{self.access_token}',
            log_path='bufferFiles/',
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=self.onopen,
            on_close=self.onclose,
            on_error=self.onerror,
            on_message=self.onmessage,
            reconnect_retry=50
        )

    def onerror(self, message):
        print('WebSocket Error:', message)

    def onclose(self, message):
        print('WebSocket Connection closed:', message)

    def onopen(self):
        print('WebSocket connected')
        try:
            self.fyers.subscribe(symbols=self.symbols, data_type='SymbolUpdate')
            print(f'Subscribed to symbols: {self.symbols}')
        except Exception as e:
            print(f'Error subscribing to symbols: {e}')

    def onmessage(self, message):
        symbol = message.get('symbol')
        if not symbol:
            print("Received message without symbol:", message)
            return

        if symbol in self.bots:
            self.bots[symbol].handle_message(message)
        else:
            print(f"Received message for unknown symbol {symbol}")

    async def init_db_pool(self):
        try:
            self.pool = await connector.create_pool(
                host='localhost',
                user='root',
                password=self.password,
                db=self.dbName,
                autocommit=True,
                minsize=5,
                maxsize=20
            )
            print("Database pool initialized successfully")
        except Exception as e:
            print(f"Error initializing database pool: {e}")

    async def run_bot(self, symbol):
        bot = self.bots[symbol]
        task = asyncio.create_task(bot.start())
        print(f"Started bot for {symbol}")
        return [task]

    async def run_all(self):
        await self.init_db_pool()
        for symbol in self.symbols:
            self.bots[symbol] = stockbot.StockBot(symbol, self.symbolData[symbol], self.pool, self.ofyers)

        for symbol in self.symbols:
            await self.run_bot(symbol)

        # Start the WebSocket connection
        self.fyers.connect()

        await asyncio.sleep(2)

        try:
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            pass



