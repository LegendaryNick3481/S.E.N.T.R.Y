"""
Creates a database (if it doesn't exist) and a table to store all trades conducted locally by
the script. Creates a table to store the closing price of the last brick of the day as well
-----------
database name = "nsetrades"
table name = "mark6" & "dayBrickClose"
password = "tiger"
"""
import mysql.connector as connector
from SymbolData import symbolData

password = "tiger"
dbName = "nsetrades"

conn = connector.connect(
    host="localhost",
    user="root",
    password=password
)
cursor = conn.cursor()
cursor.execute("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = %s", (dbName,))
result = cursor.fetchone()

if not result:
    print(f"Database '{dbName}' does not exist. Creating...")
    cursor.execute(f"CREATE DATABASE `{dbName}`")
    print(f"Database '{dbName}' created successfully.")

cursor.close()
conn.close()

conn = connector.connect(
    host="localhost",
    user="root",
    password=password,
    database=dbName
)
curr = conn.cursor()

curr.execute("""
    CREATE TABLE IF NOT EXISTS mark6 (
        id INT AUTO_INCREMENT PRIMARY KEY,
        security VARCHAR(255) NOT NULL,
        qty INT,
        buyPrice FLOAT,
        sellPrice FLOAT,
        buytime VARCHAR(255) NOT NULL,
        selltime VARCHAR(255) NOT NULL,
        grossPnL FLOAT,
        regulatoryFees FLOAT,
        netPnL FLOAT      
    )
""")

curr.execute("""
    CREATE TABLE IF NOT EXISTS dayBrickClose (
        security VARCHAR(255) PRIMARY KEY,
        date VARCHAR(255) NULL,
        dayBrickClose FLOAT NULL     
    )
""")

curr.execute("SELECT security FROM dayBrickClose")
dbtickers = [row[0] for row in curr.fetchall()]

# To insert new tickers into the table
for ticker in symbolData.keys():
    if ticker not in dbtickers:
        print(f'{ticker} not in dayBrickClose . . . Adding it ')
        curr.execute(
            """
            INSERT INTO dayBrickClose (security, dayBrickClose)
            VALUES (%s, %s)
            """,
            (ticker, symbolData[ticker]['prevDayOpen'])
        )

# To delete unwanted/redundant tickers in the table
for ticker in dbtickers:
    if ticker not in symbolData.keys():
        curr.execute("DELETE FROM dayBrickClose WHERE security = %s",(ticker,))


conn.commit()
curr.close()
conn.close()

