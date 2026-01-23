import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import re
import os
from groq import Groq
from tools.forecaster import ChronosPipeline

st.set_page_config(page_title="Borsa Ajanı & Chatbot", page_icon="🤖", layout="wide")

try:
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    else:
        GROQ_API_KEY = ""
except FileNotFoundError:
    GROQ_API_KEY = ""

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"API Key hatası: {e}")
    st.stop()

GROQ_MODEL = "llama-3.1-8b-instant"

@st.cache_resource
def load_chronos():
    return ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype="auto"
    )

try:
    chronos = load_chronos()
except Exception as e:
    st.error(f"Chronos yükleme hatası: {e}")

st.title("🤖 AI Finans Asistanı")
st.markdown("Akıllı Analiz ve Sohbet Sistemi")

tab1, tab2 = st.tabs(["📈 Analiz & Tahmin", "💬 Canlı Sohbet"])

with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Hisse Kodu (Örn: THYAO, AAPL):", "THYAO")
        analyze_btn = st.button("Analiz Et 🚀")

    if analyze_btn:
        try:
            with st.spinner('Piyasa verileri analiz ediliyor...'):
                data = yf.download(ticker, period="1y", interval="1d", progress=False)
                if data.empty and "." not in ticker:
                    temp_ticker = ticker + ".IS"
                    data = yf.download(temp_ticker, period="1y", interval="1d", progress=False)
                    if not data.empty:
                        ticker = temp_ticker

                if len(data) == 0:
                    st.error("Veri bulunamadı!")
                else:
                    clean_data = data["Close"].dropna()
                    if isinstance(clean_data, pd.DataFrame):
                        clean_data = clean_data.iloc[:, 0]
                        
                    values = clean_data.values.flatten()
                    min_val = np.min(values)
                    max_val = np.max(values)
                    range_val = max_val - min_val if max_val != min_val else 1.0
                    scaled_values = (values - min_val) / range_val
                    tensor_data = torch.tensor(scaled_values, dtype=torch.float32)

                    forecast = chronos.predict(tensor_data, prediction_length=10)
                    median_forecast = (forecast[0].median(dim=0).values.numpy() * range_val) + min_val

                    fig, ax = plt.subplots(figsize=(10, 5))
                    last_60 = clean_data.tail(60)
                    last_date = clean_data.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10)
                    ax.plot(last_60.index, last_60.values, label='Geçmiş', color='blue')
                    ax.plot(future_dates, median_forecast, label='AI Tahmin', color='red', linestyle='--')
                    ax.set_title(f"{ticker} - 10 Günlük Tahmin")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    st.subheader("📝 Yapay Zeka Uzman Görüşü")
                    current_price = clean_data.iloc[-1].item()
                    final_forecast = median_forecast[-1]
                    degisim_orani = ((final_forecast - current_price) / current_price) * 100

                    prompt_analiz = (
                        f"Sen bir borsa uzmanısın. Şu verilere göre kısa bir yorum yap: "
                        f"Hisse: {ticker}. Şu anki fiyat: {current_price:.2f}. "
                        f"Yapay zeka tahmini (10 gün sonra): {final_forecast:.2f}. "
                        f"Beklenen değişim: %{degisim_orani:.2f}. "
                        f"Yatırımcıya ne önerirsin? (Yatırım tavsiyesi değildir de)."
                        f"Cevabın Türkçe olsun."
                    )

                    try:
                        messages = [{"role": "user", "content": prompt_analiz}]

                        resp = groq_client.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=messages,
                            max_tokens=256,
                            temperature=0.7
                        )
                        st.success(resp.choices[0].message.content)

                    except Exception as e:
                        st.warning(f"Yorum alınamadı: {e}")

        except Exception as e:
            st.error(f"Hata: {e}")

with tab2:
    st.subheader("💬 Borsa Asistanı")

    if st.button("Sohbeti Temizle 🗑️"):
        st.session_state.messages = []
        st.rerun()

    def enforce_three_sentences(text: str) -> str:
        text = (text or "").strip()
        parts = re.split(r'(?<=[.!?])\s+', text)
        parts = [p.strip() for p in parts if p.strip()]

        first_two = parts[:2]
        while len(first_two) < 2:
            first_two.append("Bu konuda net konuşmak için biraz daha bağlam gerekir.")
        return f"{first_two[0]} {first_two[1]} Yatırım tavsiyesi değildir."

    def is_trade_question(user_text: str) -> str:
        t = (user_text or "").lower()
        buy_patterns = ["almalı mıyım", "almalı miyim", "alım mı", "alim mi", "almak mantıklı mı", "almak mantikli mi"]
        sell_patterns = ["satmalı mıyım", "satmalı miyim", "satayım mı", "satayim mi", "satmak mantıklı mı", "satmak mantikli mi"]
        if any(p in t for p in buy_patterns):
            return "BUY"
        if any(p in t for p in sell_patterns):
            return "SELL"
        return ""

    def extract_ticker(user_text: str) -> str:
        if not user_text:
            return ""
        candidates = re.findall(r'\b[A-Za-z]{2,6}\b', user_text)
        return candidates[0].upper() if candidates else ""

    def ensure_series(x):
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        return x

    def compute_rsi(close: pd.Series, period: int = 14) -> float:
        close = ensure_series(close).astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if pd.notna(val) else float("nan")

    @st.cache_data(ttl=900)
    def fetch_indicators(ticker: str) -> dict:
        if not ticker:
            return {"ok": False}

        tick_to_try = [ticker]
        if "." not in ticker:
            tick_to_try.append(ticker + ".IS")

        data = pd.DataFrame()
        used = ""
        for tk in tick_to_try:
            data = yf.download(tk, period="6mo", interval="1d", progress=False)
            if not data.empty:
                used = tk
                break

        if data.empty or "Close" not in data.columns:
            return {"ok": False}

        close = ensure_series(data["Close"]).dropna()
        if len(close) < 60:
            return {"ok": False}

        price = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        rsi14 = compute_rsi(close, 14)

        return {
            "ok": True,
            "ticker": used,
            "price": price,
            "ma20": ma20,
            "ma50": ma50,
            "rsi14": rsi14
        }

    def yes_no_from_indicators(trade_type: str, ind: dict) -> str:
        tkr = ind["ticker"]
        price = ind["price"]
        ma20 = ind["ma20"]
        ma50 = ind["ma50"]
        rsi14 = ind["rsi14"]

        bullish = (ma20 > ma50) and (price > ma20)
        bearish = (ma20 < ma50) and (price < ma20)

        overbought = (not np.isnan(rsi14)) and (rsi14 >= 70)

        if trade_type == "BUY":
            if bullish and (not overbought):
                return f"EVET. {tkr} için MA20>MA50 ve fiyat MA20 üstü; RSI={rsi14:.1f} aşırı alımda değil. Yatırım tavsiyesi değildir."
            return f"HAYIR. {tkr} tarafında trend net güçlü değil veya RSI={rsi14:.1f} ile aşırı alım riski var. Yatırım tavsiyesi değildir."

        if bearish or overbought:
            return f"EVET. {tkr} tarafında trend zayıf veya RSI={rsi14:.1f} aşırı alıma işaret ediyor; risk azaltma düşünülebilir. Yatırım tavsiyesi değildir."
        return f"HAYIR. {tkr} tarafında trend belirgin negatif değil ve RSI={rsi14:.1f} aşırı alım sinyali vermiyor. Yatırım tavsiyesi değildir."

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "Sen finans odaklı bir asistansın.\n"
                    "KURALLAR:\n"
                    "1) Daima TÜRKÇE yaz.\n"
                    "2) En fazla 3 cümle yaz.\n"
                    "3) Bilmediğin sayısal veriyi uydurma.\n"
                    "4) Son cümle mutlaka: 'Yatırım tavsiyesi değildir.'\n"
                )
            }
        ]

    MAX_TURNS = 16
    if len(st.session_state.messages) > MAX_TURNS:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(MAX_TURNS - 1):]

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        trade_type = is_trade_question(prompt)
        ticker_guess = extract_ticker(prompt)

        try:
            with st.spinner("Düşünüyor..."):
                if trade_type in ["BUY", "SELL"] and ticker_guess:
                    ind = fetch_indicators(ticker_guess)
                    if ind.get("ok"):
                        response_text = yes_no_from_indicators(trade_type, ind)
                    else:
                        response_text = enforce_three_sentences(
                            "Bu hisse için veri çekemedim; doğru ticker yazdığından emin olup tekrar dene."
                        )
                else:
                    prompt_for_model = (
                        "En fazla 3 cümleyle cevap ver ve son cümle 'Yatırım tavsiyesi değildir.' olsun. "
                        "Bilmediğin sayısal veriyi uydurma. "
                        + prompt
                    )

                    resp = groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=st.session_state.messages[:-1] + [{"role": "user", "content": prompt_for_model}],
                        max_tokens=220,
                        temperature=0.7,
                        top_p=0.9
                    )
                    raw = resp.choices[0].message.content
                    response_text = enforce_three_sentences(raw)

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.chat_message("assistant").markdown(response_text)

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
