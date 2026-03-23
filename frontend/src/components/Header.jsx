import React from 'react';
import { Sun, Moon, Github } from 'lucide-react';

export default function Header({ isDarkMode, setIsDarkMode, themeClasses }) {
  return (
    <header className={`flex flex-col md:flex-row justify-between items-center gap-4 p-5 rounded-3xl border shadow-xl backdrop-blur-md transition-colors duration-500 ${themeClasses.cardBg}`}>
      <div className="flex items-center gap-3">
        <Sun className="text-yellow-400 drop-shadow-lg" size={32} />
        <h1 className={`text-2xl md:text-3xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r ${themeClasses.gradientText}`}>
          Hava Durumu Tahmini
        </h1>
      </div>

      <div className="flex items-center gap-6">
        <div 
          className={`flex items-center gap-2 p-1 rounded-full cursor-pointer border transition-colors ${isDarkMode ? 'bg-[#040d1f] border-blue-500/30' : 'bg-blue-100 border-blue-300'}`}
          onClick={() => setIsDarkMode(!isDarkMode)}
        >
          <div className={`p-1.5 rounded-full transition-all ${!isDarkMode ? 'bg-white text-yellow-500 shadow-sm' : 'text-gray-400'}`}>
            <Sun size={16} />
          </div>
          <div className={`p-1.5 rounded-full transition-all ${isDarkMode ? 'bg-blue-500 text-white shadow-lg' : 'text-blue-300'}`}>
            <Moon size={16} />
          </div>
        </div>

        <a href="https://github.com/tuanaakyildiz/TuiEvolution_Weather_Report" target="_blank" rel="noopener noreferrer" className="hover:opacity-70 transition-opacity">
          <Github size={26} className={isDarkMode ? "text-white" : "text-[#1e3a8a]"} />
        </a>

        <a href="https://tuievolution.vercel.app/" target="_blank" rel="noreferrer">
          <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-sky-400 hover:scale-105 transition-transform cursor-pointer">
            TUIEVOLUTION
          </h1>
        </a>
      </div>
    </header>
  );
}