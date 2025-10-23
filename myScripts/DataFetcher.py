from fyers_apiv3 import fyersModel
import credentials as crs
import pandas as pd
from datetime import datetime, timedelta
client_id = crs.client_id
access_token = ""
with open('bufferFiles/access_token.txt') as file:
    access_token = file.read().strip()

# Initialize the FyersModel instance with your client_id, access_token, and enable async mode
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="bufferFiles/")

filename = "bufferFiles/backtestdata.csv"
date = "2025-06-17"
data = {
    "symbol": 'NSE:BANKBARODA-EQ',
    "resolution": "1",
    "date_format": "1",
    "range_from": date,
    "range_to": date,
    "cont_flag": "1"
}

sdata = fyers.history(data=data)
for i in range(len(sdata["candles"])):
    sdata["candles"][i][0] = datetime.fromtimestamp(sdata["candles"][i][0])
df = pd.DataFrame(sdata["candles"])
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Don't limit the display width
pd.set_option('display.max_colwidth', None)
print(df)
df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
df.to_csv(filename, index=False)

if __name__ == '__main__':
    pass

