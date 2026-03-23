import React from 'react';
import { Cloud, Info, Thermometer } from 'lucide-react';
import WeatherIcon from './WeatherIcon';

export default function CurrentWeather({ 
  weatherData, loading, isLocating, isCelsius, setIsCelsius, 
  themeClasses, isDarkMode, formatTemp 
}) {
  return (
    <div className={`backdrop-blur-md border rounded-3xl p-8 shadow-xl flex-1 transition-colors duration-500 ${themeClasses.cardBg}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className={`font-extrabold text-xl tracking-tight text-transparent bg-clip-text bg-gradient-to-r ${themeClasses.gradientText}`}>
          Bugün İçin Hava Tahmini
        </h2>
        {weatherData && (
          <button 
            onClick={() => setIsCelsius(!isCelsius)}
            className="flex items-center gap-1 bg-gradient-to-r from-blue-600 to-sky-400 hover:from-blue-700 hover:to-sky-500 text-white text-sm font-bold px-3 py-1.5 rounded-full shadow-lg transition-all"
          >
            <Thermometer size={16} />
            °{isCelsius ? 'C' : 'F'}
          </button>
        )}
      </div>
      
      {loading && !isLocating ? (
        <div className="flex flex-col items-center justify-center h-48 animate-pulse">
          <Cloud className="text-blue-400 mb-3" size={48} />
          <p className="font-medium text-lg">Hava durumu çekiliyor...</p>
        </div>
      ) : weatherData ? (
        <div className="flex flex-col gap-6">
          <h3 className="text-5xl font-extrabold tracking-tighter drop-shadow-md">
            {weatherData.name}
          </h3>

          <div className="flex items-center gap-5">
            <WeatherIcon condition={weatherData.weatherMain} size={80} />
            <div>
              <span className="text-7xl font-light tracking-tighter">
                {formatTemp(weatherData.temp)}°
              </span>
              <p className={`text-xl font-medium capitalize mt-1 ${themeClasses.subText}`}>{weatherData.desc}</p>
            </div>
          </div>

          <div className={`grid grid-cols-2 gap-5 mt-6 pt-6 border-t ${isDarkMode ? 'border-[#1e3a8a]' : 'border-blue-200'}`}>
            <div className={`flex flex-col gap-4 text-[15px] ${themeClasses.subText}`}>
              <div className="flex justify-between"><span className="font-bold">Hissedilen</span> <span className="font-medium">{formatTemp(weatherData.feels_like)}°{isCelsius ? 'C' : 'F'}</span></div>
              <div className="flex justify-between"><span className="font-bold">Nem</span> <span className="font-medium">%{weatherData.humidity}</span></div>
            </div>
            <div className={`flex flex-col gap-4 text-[15px] ${themeClasses.subText}`}>
              <div className="flex justify-between"><span className="font-bold">Rüzgar</span> <span className="font-medium">{weatherData.wind} kph</span></div>
              <div className="flex justify-between"><span className="font-bold">Basınç</span> <span className="font-medium">{weatherData.pressure} hPa</span></div>
            </div>
          </div>
        </div>
      ) : (
        <div className={`flex flex-col items-center justify-center h-56 text-center ${themeClasses.subText}`}>
          <Info size={48} className="mb-4 opacity-50 text-blue-500" />
          <p className="font-medium text-lg px-4">Dünyadan bir şehir arayın veya konumunuzu kullanın.</p>
        </div>
      )}
    </div>
  );
}