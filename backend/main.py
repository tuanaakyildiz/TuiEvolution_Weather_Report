from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Frontend'in (React) bağlanabilmesi için izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kaydettiğimiz dosyaları yükle
model = joblib.load("weather_model.pkl")
scaler = joblib.load("scaler.pkl")
labels = joblib.load("labels.pkl")

class WeatherInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    precipitation: float
    cloud_cover: int  # 0: clear, 1: cloudy vb. (Sıralamaya dikkat)
    atmospheric_pressure: float
    uv_index: int
    season: int       # 0: Autumn, 1: Spring vb.
    visibility: float
    location: int     # 0: Coastal, 1: Inland vb.

@app.post("/predict")
async def predict(data: WeatherInput):
    # Gelen veriyi diziye çevir
    input_features = np.array([[
        data.temperature, data.humidity, data.wind_speed, data.precipitation,
        data.cloud_cover, data.atmospheric_pressure, data.uv_index,
        data.season, data.visibility, data.location
    ]])
    
    # Normalizasyon uygula
    scaled_features = scaler.transform(input_features)
    
    # Tahmin yap
    prediction = model.predict(scaled_features)
    result = labels[prediction[0]]
    
    return {"weather_type": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)