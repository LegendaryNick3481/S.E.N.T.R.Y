"""
Initializes/restores the choppy index by giving the last 15-20 bricks of the previous/current
trading day This is done to prevent the 45 min (3 min * 15 datapoints) delay it would take main.py to
start choppyIndex (It needs atleast 15 data points to compute choppiness). The script also restores
renko bricks of the current trading day as 'chart' data points upon restart of main.py in the event
of a crash
"""
import credentials as crs
import pandas as pd
import pickle
from fyers_apiv3 import fyersModel
from datetime import datetime, timedelta
from decimal import Decimal, getcontext


def restoreData(tic, ticdata, datefrom, dateto):
    getcontext().prec = 10  # Match precision with generate_renko
    symbol_key = tic
    ticker = symbol_key[4:]
    brickSize = Decimal(str(ticdata['brickSize']))
    timeframe = ticdata['timeframe']
    prevDayOpen = Decimal(str(ticdata['prevDayOpen']))
    today = datetime.now().strftime("%Y-%m-%d")

    currentDayOpening = 0

    def fetchDataFrame():
        initPrice = Decimal('0')
        client_id = crs.client_id

        with open('bufferFiles/access_token.txt') as file:
            access_token = file.read().strip()

        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path='bufferFiles/')

        data = {
            'symbol': symbol_key,
            'resolution': timeframe,
            'date_format': '1',
            'range_from': datefrom,
            'range_to': dateto,
            'cont_flag': '1'
        }

        sdata = fyers.history(data=data)
        for i in range(len(sdata['candles'])):
            sdata['candles'][i][0] = datetime.fromtimestamp(sdata['candles'][i][0]).strftime("%Y-%m-%d %H:%M")

        Candles = pd.DataFrame(sdata['candles'])
        Candles.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        renkobackupdf = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'direction'])

        lastDirection = None
        lastRenkoClose = None

        for _, row in Candles.iterrows():
            ts = row['timestamp']
            closePrice = Decimal(str(row['close']))

            if initPrice == 0:
                initPrice = prevDayOpen
                lastRenkoClose = initPrice

            while True:
                priceDiff = closePrice - lastRenkoClose

                if abs(priceDiff) < brickSize:
                    break

                if lastDirection is None:
                    direction = 1 if priceDiff > 0 else 0
                    renkoClose = lastRenkoClose + (brickSize if direction else -brickSize)
                    renkobackupdf.loc[len(renkobackupdf)] = {
                        'timestamp': ts,
                        'open': float(lastRenkoClose),
                        'high': float(max(lastRenkoClose, renkoClose)),
                        'low': float(min(lastRenkoClose, renkoClose)),
                        'close': float(renkoClose),
                        'direction': direction
                    }
                    lastRenkoClose = renkoClose
                    lastDirection = direction
                    continue

                if lastDirection == 1:
                    if priceDiff >= brickSize:
                        renkoClose = lastRenkoClose + brickSize
                        renkobackupdf.loc[len(renkobackupdf)] = {
                            'timestamp': ts,
                            'open': float(lastRenkoClose),
                            'high': float(renkoClose),
                            'low': float(lastRenkoClose),
                            'close': float(renkoClose),
                            'direction': 1
                        }
                        lastRenkoClose = renkoClose
                    elif priceDiff <= -2 * brickSize:
                        renkoClose = lastRenkoClose - 2 * brickSize
                        renkoOpen = lastRenkoClose - brickSize
                        renkobackupdf.loc[len(renkobackupdf)] = {
                            'timestamp': ts,
                            'open': float(renkoOpen),
                            'high': float(renkoOpen),
                            'low': float(renkoClose),
                            'close': float(renkoClose),
                            'direction': 0
                        }
                        lastRenkoClose = renkoClose
                        lastDirection = 0
                    else:
                        break

                elif lastDirection == 0:
                    if priceDiff <= -brickSize:
                        renkoClose = lastRenkoClose - brickSize
                        renkobackupdf.loc[len(renkobackupdf)] = {
                            'timestamp': ts,
                            'open': float(lastRenkoClose),
                            'high': float(lastRenkoClose),
                            'low': float(renkoClose),
                            'close': float(renkoClose),
                            'direction': 0
                        }
                        lastRenkoClose = renkoClose
                    elif priceDiff >= 2 * brickSize:
                        renkoClose = lastRenkoClose + 2 * brickSize
                        renkoOpen = lastRenkoClose + brickSize
                        renkobackupdf.loc[len(renkobackupdf)] = {
                            'timestamp': ts,
                            'open': float(renkoOpen),
                            'high': float(renkoClose),
                            'low': float(renkoOpen),
                            'close': float(renkoClose),
                            'direction': 1
                        }
                        lastRenkoClose = renkoClose
                        lastDirection = 1
                    else:
                        break

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        return renkobackupdf

    backupdf = fetchDataFrame()

    if len(backupdf) < 15:
        print(f"[{ticker}] Need atleast 15 bricks for initializing indicators [Try chaning the date]")
        exit()

    if backupdf.empty:
        print('No chart data points to restore')
    else:
        with open(f"bufferFiles/{ticker}-restoreChartValues.pkl", "wb") as f:
            bf = backupdf[backupdf['timestamp'].str.startswith(dateto)]
            pickle.dump(bf, f)
        with open(f"bufferFiles/{ticker}-restoreBricksForCI.pkl", "wb") as f:
            bff = backupdf[~backupdf['timestamp'].str.startswith(dateto)]
            pickle.dump(bff, f)
        with open(f"bufferFiles/{ticker}-getDayOpening.pkl", "wb") as f:
            bff = backupdf[backupdf['timestamp'].str.startswith(dateto) == False ]
            pickle.dump(bff, f)



if __name__ == '__main__':
    restoreData('NSE:JMFINANCIL-EQ',
                {'brickSize':0.5,'timeframe':1,'prevDayOpen':100.50},
                '2025-05-02',
                '2025-05-30')







