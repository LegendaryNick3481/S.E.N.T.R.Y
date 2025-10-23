import createDB
import asyncio
import pickle
import MasterBot as master_module
import VWAPStockBot as bot
import RestoreData as restore2
import mysql.connector as connector
from SymbolData import symbolData
from gui import BotGUI  # GUI class in a separate file (optional)
import threading
import time
import tkinter as tk

def load_symbol_data_from_db():
    password = "tiger"
    dbName = "nsetrades"

    conn = connector.connect(
        host="localhost",
        user="root",
        password=password,
        database=dbName
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dayBrickClose")
    result = cursor.fetchall()

    for i in range(len(result)):
        t = result[i][0]
        symbolData[t]['prevDayOpen'] = result[i][2]

    print('symbolData initialized with previous day brick close from the Database')
    cursor.close()
    conn.close()

def restore_all_data():
    for data in symbolData:
        restore2.restoreData(
            data,
            symbolData[data],
            '2025-06-25',
            '2025-06-30'
        )

if __name__ == "__main__":
    load_symbol_data_from_db()
    restore_all_data()

    master_bot = master_module.MasterBot(symbolData)

    loop = asyncio.new_event_loop()

    def start_async():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(master_bot.run_all())

    threading.Thread(target=start_async, daemon=True).start()

    time.sleep(1)  # give some time for bots to be ready

    # Launch GUI in main thread
    gui = BotGUI(master_bot, loop)
    gui.run()
