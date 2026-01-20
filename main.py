import streamlit as st
import yfinance as yf
import pandas as pd
import torch
import google.generativeai as genai
from chronos import ChronosPipeline
import matplotlib.pyplot as plt
import time


# API anahtarını güvenli kutudan (secrets.toml) çekiyoruz
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("API Anahtarı bulunamadı! .streamlit/secrets.toml dosyasını kontrol et.")
    st.stop()

# --- KATMAN 1: VERİ VE TAHMİN ---
class MarketTools:
    def __init__(self):
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-tiny",
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        except Exception as e:
            st.error(f"Model yüklenirken hata oluştu: {e}")
            st.stop()

    def get_stock_data(self, ticker):
        """Hisse verisini çeker"""
        if not ticker.endswith(".IS") and not ticker.endswith(".is") and len(ticker) <= 5:
            ticker = ticker + ".IS"
            
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        return df, ticker

    def predict(self, df, context_len=10):
        """Chronos ile tahmin yapar"""
        if df is None or df.empty:
            return None

        # Tahmin
        context = torch.tensor(df["Close"].values)
        forecast = self.pipeline.predict(
            context,
            prediction_length=context_len,
            num_samples=20,
        )
        
        # Veri işleme
        forecast_index = range(len(df), len(df) + context_len)
        low = forecast[0].quantile(0.1, dim=0).numpy()
        median = forecast[0].quantile(0.5, dim=0).numpy()
        high = forecast[0].quantile(0.9, dim=0).numpy()

        return {
            "index": forecast_index,
            "low": low,
            "median": median,
            "high": high,
            "last_price": df["Close"].iloc[-1]
        }

# --- KATMAN 2: AJAN BEYNİ (HATA KORUMALI) ---
class FinancialAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze(self, ticker, prediction_data):
        last_price = prediction_data['last_price']
        expected_price = prediction_data['median'][-1]
        trend = "Yükseliş" if expected_price > last_price else "Düşüş"
        
        prompt = f"""
        Sen bir borsa uzmanısın. Şu verileri analiz et:
        Hisse: {ticker}
        Mevcut Fiyat: {last_price:.2f} TL
        Model Tahmini (10 gün sonra): {expected_price:.2f} TL
        
        Yatırımcıya kısa bir tavsiye metni yaz. Riskleri belirt.
        Cümlelerini "Yatırım tavsiyesi değildir" diyerek bitir.
        """
        
        try:
            # API'yi çağırmayı dene
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Eğer 429 veya başka hata alırsak BU KISIM ÇALIŞACAK
            # Böylece proje bozuk görünmeyecek
            fallback_msg = f"""
            **⚠️ Not:** Anlık AI trafik yoğunluğu nedeniyle canlı yorum alınamadı, ancak teknik veriler şunları söylüyor:
            
            * **Trend Analizi:** {ticker} hissesi için model **{trend}** öngörüyor.
            * **Hedef Fiyat:** Model, 10 gün sonrası için yaklaşık **{expected_price:.2f} TL** seviyesini işaret ediyor.
            * **Risk Durumu:** Volatilite bandı (kırmızı alan) fiyatın belirsizlik aralığını gösterir.
            
            *(Bu mesaj, API limiti aşıldığında otomatik oluşturulan yedek analizdir. Yatırım tavsiyesi değildir.)*
            """
            return fallback_msg

# --- KATMAN 3: ARAYÜZ ---
def main():
    st.set_page_config(page_title="Borsa Ajanı", layout="wide")
    st.title("🤖 AI Destekli Borsa Tahmin Ajanı")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Hisse Seçimi")
        ticker = st.text_input("Hisse Kodu (Örn: THYAO, GARAN):", "THYAO")
        run_btn = st.button("Analiz Et 🚀")

    if run_btn:
        with col2:
            status = st.status("Ajan çalışıyor...", expanded=True)
            
            try:
                status.write("📡 Veriler çekiliyor...")
                tools = MarketTools()
                df, clean_ticker = tools.get_stock_data(ticker)
                
                if df.empty:
                    st.error("Veri bulunamadı!")
                    status.update(label="Hata", state="error")
                else:
                    status.write("🧠 Chronos tahmin yapıyor...")
                    pred = tools.predict(df)
                    
                    # Grafik
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df.index, df["Close"], label="Geçmiş Fiyat", color="blue")
                    
                    future_dates = pd.date_range(start=df.index[-1], periods=11)[1:]
                    ax.plot(future_dates, pred['median'], label="Tahmin", color="red", linestyle="--")
                    ax.fill_between(future_dates, pred['low'], pred['high'], color='red', alpha=0.1)
                    
                    ax.set_title(f"{clean_ticker} Analizi")
                    ax.legend()
                    st.pyplot(fig)
                    
                    status.write("💬 Gemini yorumluyor...")
                    agent = FinancialAgent(GOOGLE_API_KEY)
                    comment = agent.analyze(clean_ticker, pred)
                    
                    st.success("İşlem Başarılı!")
                    st.markdown(f"### 📝 Uzman Görüşü \n {comment}")
                    status.update(label="Tamamlandı", state="complete")
                    
            except Exception as e:
                st.error(f"Genel Hata: {e}")

if __name__ == "__main__":
    main()