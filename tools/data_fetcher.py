import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker: str, period="3mo"):
    """
    Belirtilen hisse senedi için son 3 aylık OHLC verilerini çeker.
    Örn: ticker='THYAO.IS' (BIST hisseleri için .IS eklenmeli)
    """
    # Türk hisseleri için sonuna .IS ekleme kontrolü
    if not ticker.endswith(".IS") and not ticker.endswith(".is"):
         # Kullanıcı global hisse (AAPL) mi yoksa BIST mi istiyor diye varsayım yapmak gerekebilir.
         # Şimdilik BIST varsayalım veya kullanıcıdan tam kod isteyelim.
         pass 

    print(f"Veri çekiliyor: {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        return None
    
    # Chronos için sadece gerekli sütunları ve formatı hazırlayalım
    # Genelde 'Close' veya 'Open' üzerinden tahmin yapılır ama Chronos full time series sever.
    return df[['Open', 'High', 'Low', 'Close']]