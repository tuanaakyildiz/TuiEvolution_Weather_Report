import React from 'react';
import { Search, MapPin, Cloud, Loader2 } from 'lucide-react';

export default function SearchBar({
  searchQuery, setSearchQuery, showDropdown, setShowDropdown, 
  handleLocationClick, isLocating, isSearchingGlobal, 
  globalCities, filteredImportantCities, fetchWeatherByCityName, 
  fetchWeatherByCoords, themeClasses, isDarkMode
}) {
  return (
    <div className={`relative backdrop-blur-md border rounded-3xl p-4 shadow-xl transition-colors duration-500 z-50 ${themeClasses.cardBg}`}>
      <div className={`flex items-center rounded-2xl px-4 py-3 border transition-colors ${themeClasses.inputBg}`}>
        <Search size={20} className={isDarkMode ? "text-blue-400 mr-3" : "text-blue-600 mr-3"} />
        <input 
          type="text" 
          placeholder="Dünyadan bir şehir arayın..." 
          className="bg-transparent border-none outline-none w-full font-medium"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onFocus={() => setShowDropdown(true)}
        />
      </div>

      {showDropdown && (
        <div className={`absolute top-full left-0 right-0 mt-3 border rounded-2xl overflow-hidden shadow-2xl ${themeClasses.dropdownBg}`}>
          <button 
            onClick={handleLocationClick}
            disabled={isLocating}
            className={`w-full text-left px-5 py-4 flex items-center gap-3 border-b border-opacity-20 font-bold ${isDarkMode ? "text-sky-300" : "text-blue-700"} ${themeClasses.hoverBg} ${isLocating ? "opacity-70 cursor-not-allowed" : ""}`}
          >
            {isLocating ? (
              <><Loader2 size={20} className="text-blue-500 animate-spin" /> Konumunuz aranıyor...</>
            ) : (
              <><MapPin size={20} className="text-blue-500" /> Konumuma Göre Ara</>
            )}
          </button>
          
          <div className="max-h-64 overflow-y-auto scrollbar-thin scrollbar-thumb-blue-500/20">
            {searchQuery.length < 3 ? (
              filteredImportantCities.length > 0 ? (
                filteredImportantCities.map((city, index) => (
                  <button
                    key={index}
                    onClick={() => fetchWeatherByCityName(city)}
                    className={`w-full text-left px-5 py-3 font-medium flex justify-between items-center ${themeClasses.hoverBg}`}
                  >
                    <span>{city}</span>
                  </button>
                ))
              ) : (
                <div className="px-5 py-4 text-sm opacity-70">Dünyada aramak için yazmaya devam edin...</div>
              )
            ) : (
              isSearchingGlobal ? (
                <div className="px-5 py-4 text-sm font-medium text-blue-400 flex items-center gap-2">
                  <Cloud size={16} className="animate-pulse" /> Dünyada aranıyor...
                </div>
              ) : globalCities.length > 0 ? (
                globalCities.map((city, index) => (
                  <button
                    key={index}
                    onClick={() => fetchWeatherByCoords(city.lat, city.lon, city.name)}
                    className={`w-full text-left px-5 py-3 font-medium flex justify-between items-center ${themeClasses.hoverBg}`}
                  >
                    <span>{city.name} {city.state && <span className="text-xs opacity-60 ml-1">({city.state})</span>}</span>
                    <span className="text-xs font-bold bg-blue-500/20 text-blue-300 px-2 py-1 rounded-md">{city.country}</span>
                  </button>
                ))
              ) : (
                <div className="px-5 py-4 text-sm opacity-70">Dünya genelinde eşleşen şehir bulunamadı.</div>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}