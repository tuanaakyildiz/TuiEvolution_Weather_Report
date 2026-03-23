import os
import requests
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# 1. Çevresel Değişkenleri Yükle
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Sabit Dosya Yolları
MODEL_PATH = "weather_model.pkl"
SCALER_PATH = "scaler.pkl"
LABELS_PATH = "labels.pkl"

# --- YARDIMCI FONKSİYONLAR ---

def get_training_data():
    """
    OpenWeather API'den 5 günlük/3 saatlik tahmin verisini çeker.
    (Ücretsiz API ile uyumlu versiyon)
    """
    all_data = []
    lat, lon = 41.0082, 28.9784 # İstanbul Koordinatları
    
    # ÜCRETSİZ ENDPOINT: 'forecast' kullanıyoruz
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        # Hata varsa (401, 404 vs) fırlat
        response.raise_for_status()
        
        res = response.json()
        
        if 'list' in res:
            for item in res['list']:
                all_data.append({
                    "temp": item['main']['temp'],
                    "humidity": item['main']['humidity'],
                    "wind": item['wind']['speed'],
                    "clouds": item['clouds']['all'],
                    "pressure": item['main']['pressure'],
                    "weather_main": item['weather'][0]['main']
                })
        else:
            print("API yanıtı beklenen formatta değil:", res)

    except Exception as e:
        print(f"Veri çekme hatası: {e}")
            
    return pd.DataFrame(all_data)

def train_model():
    """Modeli eğitir ve diske kaydeder."""
    if not API_KEY:
        print("HATA: API_KEY .env dosyasında bulunamadı!")
        return

    print(">>> Veriler çekiliyor (Forecast API)...")
    df = get_training_data()
    
    if df.empty:
        print("HATA: Veri seti boş. Model eğitilemedi.")
        print("OLASI SEBEP: API Key geçersiz veya henüz aktifleşmedi.")
        return

    # Etiketleri Hazırla
    df['weather_code'] = df['weather_main'].astype('category').cat.codes
    labels = df['weather_main'].astype('category').cat.categories.tolist()
    
    X = df[['temp', 'humidity', 'wind', 'clouds', 'pressure']]
    y = df['weather_code']
    
    # Normalizasyon
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ANN Modeli (MLP)
    mlp = MLPClassifier(max_iter=2000, random_state=42)
    mlp.fit(X_scaled, y)
    
    # Kayıt İşlemleri
    joblib.dump(mlp, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(labels, LABELS_PATH)
    print(f">>> Model başarıyla eğitildi. Sınıflar: {labels}")

# --- FASTAPI YAŞAM DÖNGÜSÜ (LIFESPAN) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uygulama başlarken:
    print(">>> Sistem başlatılıyor...")
    train_model() 
    yield
    # Uygulama kapanırken:
    print(">>> Sistem kapatılıyor...")

# FastAPI Uygulaması
app = FastAPI(lifespan=lifespan)

# CORS Ayarları (Frontend'den gelen isteklere izin ver)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Güvenlik için production'da spesifik domain yazılır
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINT'LER ---

@app.post("/predict")
async def predict(data: dict):
    try:
        # Modelleri yükle
        if not os.path.exists(MODEL_PATH):
             return {"error": "Model henüz eğitilmedi, lütfen bekleyin veya logları kontrol edin."}

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        labels = joblib.load(LABELS_PATH)
        
        # Gelen veriyi formatla
        input_array = np.array([[
            data['temp'], 
            data['humidity'], 
            data['wind'], 
            data['clouds'], 
            data['pressure']
        ]])
        
        # Tahmin
        scaled = scaler.transform(input_array)
        res = model.predict(scaled)
        
        return {"prediction": labels[res[0]]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)