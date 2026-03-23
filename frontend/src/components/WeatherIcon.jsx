import React from 'react';
// Yeni ikonları ekledik: CloudLightning ve CloudDrizzle
import { Sun, Cloud, CloudRain, CloudLightning, Snowflake, CloudDrizzle } from 'lucide-react';

export default function WeatherIcon({ condition, size = 48 }) {
  // Arka plandan (backend) gelen veriyi küçük harfe çevirip kontrol ediyoruz
  switch (condition?.toLowerCase()) {
    case 'clear': return <Sun size={size} className="text-yellow-400 drop-shadow-md" />;
    case 'clouds': return <Cloud size={size} className="text-gray-300 drop-shadow-md" />;
    case 'rain': return <CloudRain size={size} className="text-blue-400 drop-shadow-md" />;
    
    // --- YENİ EKLENEN DURUMLAR ---
    case 'drizzle': return <CloudDrizzle size={size} className="text-blue-300 drop-shadow-md" />;
    case 'thunderstorm': return <CloudLightning size={size} className="text-purple-500 drop-shadow-md" />;
    
    case 'snow': return <Snowflake size={size} className="text-sky-200 drop-shadow-md" />;
    default: return <Sun size={size} className="text-yellow-400 drop-shadow-md" />;
  }
}