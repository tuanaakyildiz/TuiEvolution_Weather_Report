import os
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel

# --- 1. AYARLAR VE ÇEVRESEL DEĞİŞKENLER ---
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

global_model = None
global_scaler = None
global_labels = None

# --- 2. VERİ DOĞRULAMA (PYDANTIC) ---
class WeatherInput(BaseModel):
    temp: float
    humidity: float
    wind: float
    clouds: int
    pressure: float

# --- 3. DİNAMİK EĞİTİM FONKSİYONLARI (IN-MEMORY) ---
def get_training_data():
    """
    Modelin her hava durumunu (Clear, Snow vb.) kesinlikle görebilmesi için
    kapsamlı, meteorolojik mantığa dayalı sentetik bir veri seti üretir.
    """
    data = [
        # CLEAR (GÜNEŞLİ/AÇIK) - Genelde bulutsuz, nem dengeli, basınç yüksek olur
        {"temp": 25.0, "humidity": 40.0, "wind": 3.0, "clouds": 0, "pressure": 1015.0, "weather_main": "Clear"},
        {"temp": 30.0, "humidity": 30.0, "wind": 5.0, "clouds": 5, "pressure": 1012.0, "weather_main": "Clear"},
        {"temp": 15.0, "humidity": 50.0, "wind": 2.0, "clouds": 10, "pressure": 1020.0, "weather_main": "Clear"},
        {"temp": 35.0, "humidity": 20.0, "wind": 1.0, "clouds": 0, "pressure": 1010.0, "weather_main": "Clear"},
        
        # CLOUDS (BULUTLU) - Bulutluluk yüksek, nem orta/yüksek
        {"temp": 18.0, "humidity": 65.0, "wind": 4.0, "clouds": 80, "pressure": 1008.0, "weather_main": "Clouds"},
        {"temp": 12.0, "humidity": 75.0, "wind": 6.0, "clouds": 100, "pressure": 1005.0, "weather_main": "Clouds"},
        {"temp": 22.0, "humidity": 60.0, "wind": 3.0, "clouds": 60, "pressure": 1011.0, "weather_main": "Clouds"},
        {"temp": 10.0, "humidity": 80.0, "wind": 7.0, "clouds": 90, "pressure": 1002.0, "weather_main": "Clouds"},
        
        # RAIN / DRIZZLE (YAĞMURLU) - Nem çok yüksek, bulutluluk tavan, basınç düşük
        {"temp": 15.0, "humidity": 90.0, "wind": 5.0, "clouds": 100, "pressure": 1000.0, "weather_main": "Rain"},
        {"temp": 10.0, "humidity": 95.0, "wind": 8.0, "clouds": 100, "pressure": 995.0, "weather_main": "Rain"},
        {"temp": 20.0, "humidity": 85.0, "wind": 4.0, "clouds": 90, "pressure": 1005.0, "weather_main": "Rain"},
        {"temp": 12.0, "humidity": 88.0, "wind": 6.0, "clouds": 95, "pressure": 998.0, "weather_main": "Drizzle"},
        
        # THUNDERSTORM (FIRTINA/GÖK GÜRÜLTÜSÜ) - Basınç çok düşük, rüzgar şiddetli
        {"temp": 18.0, "humidity": 90.0, "wind": 15.0, "clouds": 100, "pressure": 990.0, "weather_main": "Thunderstorm"},
        {"temp": 22.0, "humidity": 85.0, "wind": 20.0, "clouds": 100, "pressure": 985.0, "weather_main": "Thunderstorm"},
        
        # SNOW (KARLI) - Sıcaklık sıfır veya altı, nem yüksek
        {"temp": -2.0, "humidity": 80.0, "wind": 4.0, "clouds": 100, "pressure": 1005.0, "weather_main": "Snow"},
        {"temp": -5.0, "humidity": 75.0, "wind": 6.0, "clouds": 90, "pressure": 1010.0, "weather_main": "Snow"},
        {"temp": 1.0, "humidity": 85.0, "wind": 3.0, "clouds": 100, "pressure": 1002.0, "weather_main": "Snow"},
        {"temp": -10.0, "humidity": 70.0, "wind": 10.0, "clouds": 80, "pressure": 1015.0, "weather_main": "Snow"}
    ]
    
    # Modele hacim kazandırmak için veriyi çoğaltıyoruz
    large_data = data * 20 
    return pd.DataFrame(large_data)

def train_model_in_memory():
    global global_model, global_scaler, global_labels

    print(">>> AI Modeli için kapsamlı meteorolojik veriler yükleniyor...")
    df = get_training_data()

    # Etiketleri Hazırla
    df['weather_code'] = df['weather_main'].astype('category').cat.codes
    global_labels = df['weather_main'].astype('category').cat.categories.tolist()
    
    X = df[['temp', 'humidity', 'wind', 'clouds', 'pressure']]
    y = df['weather_code']
    
    # Normalizasyon
    global_scaler = MinMaxScaler()
    X_scaled = global_scaler.fit_transform(X)
    
    # ANN Modeli Eğitimi
    global_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    global_model.fit(X_scaled, y)
    
    print(f">>> Model başarıyla eğitildi ve RAM'e alındı! Sınıflar: {global_labels}")

# --- 4. FASTAPI YAŞAM DÖNGÜSÜ ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    train_model_in_memory()
    yield
    print(">>> Sistem kapatılıyor, RAM temizleniyor...")

# --- 5. FASTAPI UYGULAMASI VE CORS ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 6. TAHMİN (PREDICT) ENDPOINT'İ ---
@app.post("/predict")
async def predict(data: WeatherInput):
    global global_model, global_scaler, global_labels
    
    try:
        if global_model is None or global_scaler is None:
             return {"error": "Model şu an RAM'de hazır değil, lütfen birazdan tekrar deneyin."}

        input_array = np.array([[
            data.temp, 
            data.humidity, 
            data.wind, 
            data.clouds, 
            data.pressure
        ]])
        
        scaled = global_scaler.transform(input_array)
        res = global_model.predict(scaled)
        
        return {"prediction": global_labels[res[0]]}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)