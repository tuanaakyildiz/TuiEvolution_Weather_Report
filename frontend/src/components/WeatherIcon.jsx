import React from 'react';
import { Sun, Cloud, CloudRain, CloudLightning, Snowflake } from 'lucide-react';

export default function WeatherIcon({ condition, size = 48 }) {
  switch (condition?.toLowerCase()) {
    case 'clear': return <Sun size={size} className="text-yellow-400 drop-shadow-md" />;
    case 'clouds': return <Cloud size={size} className="text-gray-300 drop-shadow-md" />;
    case 'rain': 
    case 'drizzle': return <CloudRain size={size} className="text-blue-400 drop-shadow-md" />;
    case 'thunderstorm': return <CloudLightning size={size} className="text-blue-600 drop-shadow-md" />;
    case 'snow': return <Snowflake size={size} className="text-sky-200 drop-shadow-md" />;
    default: return <Sun size={size} className="text-yellow-400 drop-shadow-md" />;
  }
}