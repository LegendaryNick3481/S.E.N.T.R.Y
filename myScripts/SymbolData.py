"""
Initialize the prevDayClose with the previous trading day's last brick close once. Subsequent day's brick close will
be saved automatically by the script
"""

symbolData = {
    "NSE:JMFINANCIL-EQ":
        {
            "brickSize": 0.5,
            "timeframe": 1,
            "quantity": 1,
            "prevDayOpen": 150.50
        },
}

# "NSE:POONAWALLA-EQ":
# {
#     "brickSize": 0.5,
#     "timeframe": 3,
#     "quantity": 3,
#     "prevDayOpen": 444.00
# },
#
# "NSE:ISHANCH-EQ":
# {
#     "brickSize": 0.25,
#     "timeframe": 3,
#     "quantity": 22,
#     "prevDayOpen": 52.50
# },
#
# "NSE:PRABHA-EQ":
# {
#     "brickSize": 1,
#     "timeframe": 3,
#     "quantity": 4,
#     "prevDayOpen": 258.00
# },
# "NSE:RBLBANK-EQ":
# {
#     "brickSize": 1,
#     "timeframe": 3,
#     "quantity": 5,
#     "prevDayOpen": 237.00
# },
# "NSE:HFCL-EQ":
# {
#     "brickSize": 0.25,
#     "timeframe": 3,
#     "quantity": 5,
#     "prevDayOpen": 84.00
# },



