/** @type {import('tailwindcss').Config} */
export default {
    content: [
      "./index.html",
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class', // Karanlık mod desteği için
    theme: {
      extend: {
        colors: {
          textprimary: "#ffffff",
          accent: "#a855f7", // Tuievolution logosu için mor tonu
        }
      },
    },
    plugins: [],
  }