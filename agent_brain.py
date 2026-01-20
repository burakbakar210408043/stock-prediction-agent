import google.generativeai as genai
import os
from tools.data_fetcher import get_stock_data
from tools.forecaster import predict_stock_trend

# API Key yapılandırması
# os.environ["GOOGLE_API_KEY"] = "SENIN_API_KEYIN"

def analyze_stock(user_message):
    """
    Kullanıcı mesajını alır, gerekirse araçları çağırır ve yorum yapar.
    """
    model = genai.GenerativeModel('gemini-1.5-flash') # Hızlı ve tool calling destekli
    
    # Tool'ları modele tanıtma (Function Calling yapısı)
    # Not: Gemini SDK'sında tools parametresi ile bu fonksiyonlar bağlanır.
    # Basitlik adına burada mantığı "ReAct" (Reasoning + Acting) olarak simüle ediyorum.
    
    # 1. LLM hisse kodunu bulsun
    prompt = f"""
    Sen uzman bir borsa asistanısın. Kullanıcının sorduğu hisse senedini analiz etmelisin.
    Kullanıcı mesajı: "{user_message}"
    
    Adımlar:
    1. Mesajdan hisse kodunu çıkar (Örn: THYAO ise THYAO.IS yap).
    2. Son 3 aylık veriyi çek.
    3. Chronos modelini kullanarak tahmin yap.
    4. Sonucu finansal okuryazarlığa uygun şekilde yorumla. Yatırım tavsiyesi değildir uyarısını ekle.
    """
    
    # Burada normalde "Function Calling" ile otomatik tetikleme yapılır.
    # Ancak manuel akış şöyle olur:
    
    # Örnek statik akış (Kodun çalışması için):
    ticker = "THYAO.IS" # Bu kısmı LLM'den çektireceğiz
    data = get_stock_data(ticker)
    
    if data is not None:
        prediction = predict_stock_trend(data)
        
        final_prompt = f"""
        Hisse: {ticker}
        Mevcut Son Fiyat: {data['Close'].iloc[-1]}
        Model Tahmini (10 gün sonrası için):
        - Beklenen: {prediction['expected_price']}
        - Alt Sınır: {prediction['lower_bound']}
        - Üst Sınır: {prediction['upper_bound']}
        - Trend: {prediction['trend']}
        
        Bu verileri kullanarak kullanıcıya profesyonel bir yorum yaz.
        """
        response = model.generate_content(final_prompt)
        return response.text
    else:
        return "Hisse verisi çekilemedi."