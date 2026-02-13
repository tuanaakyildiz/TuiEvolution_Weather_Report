import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_KEY = import.meta.env.VITE_OPENWEATHER_API_KEY;
const BACKEND_URL = import.meta.env.VITE_API_URL;

function App() {
  const [weatherData, setWeatherData] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);

  const handlePredict = () => {
    setLoading(true);
    navigator.geolocation.getCurrentPosition(async (pos) => {
      try {
        const { latitude, longitude } = pos.coords;
        
        // 1. Güncel veriyi çek
        const res = await axios.get(
          `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${API_KEY}&units=metric`
        );
        
        const currentFields = {
          temp: res.data.main.temp,
          humidity: res.data.main.humidity,
          wind: res.data.wind.speed,
          clouds: res.data.clouds.all,
          pressure: res.data.main.pressure
        };
        
        setWeatherData(currentFields);

        // 2. Backend'e gönder
        const predictRes = await axios.post(`${BACKEND_URL}/predict`, currentFields);
        setPrediction(predictRes.data.prediction);
      } catch (err) {
        console.error("Hata:", err);
        alert("Veri çekilemedi veya Backend çalışmıyor.");
      } finally {
        setLoading(false);
      }
    });
  };

  const getBgClass = () => {
    if (!prediction) return "default-bg";
    const p = prediction.toLowerCase();
    if (p.includes("rain")) return "rainy-bg";
    if (p.includes("cloud")) return "cloudy-bg";
    if (p.includes("clear")) return "sunny-bg";
    return "default-bg";
  };

  return (
    <div className={`app-container ${getBgClass()}`}>
      <div className="glass-card">
        <h2>Weather AI Predictor</h2>
        <button className="predict-btn" onClick={handlePredict} disabled={loading}>
          {loading ? "Hesaplanıyor..." : "Konum Al ve Tahmin Et"}
        </button>

        {prediction && (
          <div className="info">
            <h1 className="prediction-text">{prediction}</h1>
            <p>Sıcaklık: {weatherData?.temp}°C</p>
            <p>Nem: %{weatherData?.humidity}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;