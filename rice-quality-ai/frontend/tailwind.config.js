/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        grain: {
          50: '#fbf8f3',
          100: '#f5eee3',
          200: '#ebdcbe',
          300: '#dec393',
          400: '#d0a66b',
          500: '#c58d4a',
          600: '#b2743c',
          700: '#945a33',
          800: '#79482e',
          900: '#643d29',
        },
        emerald: {
          DEFAULT: '#10b981',
        },
        slate: {
          850: '#172033',
          950: '#0b0f19',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
