"""
Stock tickers for the trading system
"""

TICKERS = [
    "HINDZINC",      # Hindustan Zinc Ltd
    "MANKIND",       # Mankind Pharma Ltd
    "INDUSTOWER",    # Indus Towers Ltd
    "DEEPINDS",      # Deep Industries Ltd
    "FMGOETZE",      # Federal-Mogul Goetze (India) Ltd
    "DBCORP",        # D B Corp Ltd
    "TALBROAUTO",    # Talbros Automotive Components Ltd
    "JAGSNPHARM",    # Jagsonpal Pharmaceuticals Ltd
    "SHREEJI",       # Shreeji Translogistics Ltd
    "RAJOOENG",      # Rajoo Engineers Ltd
    "MONARCH",       # Monarch Networth Capital Ltd
    "RESPONIND",     # Responsive Industries Ltd
    "BLS",           # BLS International Services Ltd
    "EIHAHOTELS",    # EIH Associated Hotels Ltd
    "SAKSOFT"        # Saksoft Ltd
]

def get_tickers():
    """Get list of tickers"""
    return TICKERS

def get_fyers_symbol(ticker):
    """Convert ticker to Fyers format: NSE:SYMBOL-EQ"""
    return f"NSE:{ticker}-EQ"
