import torch
from chronos import ChronosPipeline # Chronos kütüphanesi (github reposundan kurulmalı)

# Modeli global olarak bir kez yüklemek performans için iyidir
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny", # Test için 'tiny', prodüksiyon için 'small' veya 'base'
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

def predict_stock_trend(df, prediction_length=10):
    """
    Verilen dataframe'i alır ve Chronos ile gelecek 10 günün tahmin aralığını döndürür.
    """
    # Veriyi Chronos'un istediği tensör formatına çevirme
    context = torch.tensor(df["Close"].values)
    
    # Tahmin yap
    forecast = pipeline.predict(
        context, 
        prediction_length=prediction_length,
        num_samples=20
    )
    
    # Çıktıyı insan (ve LLM) tarafından okunabilir formata sok
    # Median (beklenen) ve %90 güven aralığını alalım
    low = forecast[0].quantile(0.1).item()
    median = forecast[0].quantile(0.5).item()
    high = forecast[0].quantile(0.9).item()
    
    return {
        "expected_price": round(median, 2),
        "lower_bound": round(low, 2),
        "upper_bound": round(high, 2),
        "trend": "Yükseliş" if median > df["Close"].iloc[-1] else "Düşüş"
    }