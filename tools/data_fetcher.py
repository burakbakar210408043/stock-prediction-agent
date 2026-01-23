import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker: str, period="3mo"):
    """
    Belirtilen hisse senedi için son 3 aylık OHLC verilerini çeker.
    Örn: ticker='THYAO.IS' (BIST hisseleri için .IS eklenmeli)
    """
    
    if not ticker.endswith(".IS") and not ticker.endswith(".is"):
     
         pass 

    print(f"Veri çekiliyor: {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        return None
 
    return df[['Open', 'High', 'Low', 'Close']]
