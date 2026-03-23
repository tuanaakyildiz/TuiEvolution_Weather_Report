import React from 'react';
import { CloudRain } from 'lucide-react';
import WeatherIcon from './WeatherIcon';

export default function Forecast({ forecastData, themeClasses, isDarkMode, formatTemp }) {
  return (
    <div className={`lg:col-span-2 backdrop-blur-md border rounded-3xl p-8 shadow-xl flex flex-col transition-colors duration-500 ${themeClasses.cardBg}`}>
      <h2 className={`font-extrabold text-3xl mb-8 text-center pb-6 border-b-2 tracking-tight transition-all duration-500 text-transparent bg-clip-text bg-gradient-to-r ${themeClasses.gradientText} ${isDarkMode ? 'border-[#1e3a8a]' : 'border-blue-200'}`}>
        5 Günlük Hava Durumu
      </h2>
      
      <div className="flex-1 flex items-center justify-center">
        {forecastData.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-5 w-full">
            {forecastData.map((day, idx) => {
              const date = new Date(day.dt * 1000);
              const dayName = date.toLocaleDateString('tr-TR', { weekday: 'long' });
              const dayNum = date.toLocaleDateString('tr-TR', { day: 'numeric', month: 'short' });

              return (
                <div key={idx} className={`flex flex-col items-center justify-center rounded-2xl p-6 transition-all border shadow-sm ${isDarkMode ? 'bg-[#040d1f]/60 border-transparent hover:border-blue-500/50 hover:bg-[#040d1f]/90' : 'bg-blue-50/50 border-blue-100 hover:border-blue-300 hover:shadow-md hover:bg-white'}`}>
                  <span className={`text-sm font-bold mb-1 opacity-80 ${themeClasses.subText}`}>{dayNum}</span>
                  <span className="font-extrabold mb-5 text-center text-lg">{dayName}</span>
                  
                  <WeatherIcon condition={day.weather[0].main} size={56} />
                  
                  <span className={`text-sm mt-4 font-medium capitalize text-center h-10 flex items-center ${themeClasses.subText}`}>
                    {day.weather[0].description}
                  </span>
                  
                  <div className="mt-3 flex gap-2 font-black text-2xl">
                    <span>{formatTemp(day.main.temp_max)}°</span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className={`flex flex-col items-center text-center opacity-60 ${themeClasses.subText}`}>
            <CloudRain size={64} className="mb-4" />
            <p className="font-medium text-xl">Tahmin verileri bekleniyor...</p>
          </div>
        )}
      </div>
    </div>
  );
}