/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: '#285943',
          light: '#E8F0EA',
          dark: '#1E4534',
        },
        sage: {
          DEFAULT: '#6F8F78',
          light: '#E8F0EA',
          dark: '#5A7562',
        },
        surface: {
          DEFAULT: '#FFFFFF',
          page: '#F7F5EF',
          subtle: '#F2F5F2',
        },
        text: {
          primary: '#26332B',
          secondary: '#66736B',
          muted: '#8A948E',
        },
        border: {
          DEFAULT: '#DCE4DE',
          dark: '#B8C4BB',
        },
        success: {
          DEFAULT: '#3F7D58',
          bg: '#EAF3ED',
          text: '#2D5A3F',
        },
        warning: {
          DEFAULT: '#A97832',
          bg: '#F8F0E2',
          text: '#6F5427',
        },
        error: {
          DEFAULT: '#B4534B',
          bg: '#F8EAE8',
          text: '#7A2E28',
        },
        info: {
          DEFAULT: '#527A8A',
          bg: '#EAF1F4',
          text: '#3A5A68',
        },
        notrice: {
          DEFAULT: '#6F5427',
          bg: '#F8F0E2',
          border: '#E5D5B7',
          text: '#6F5427',
        },
        grain: {
          1: '#2F6F5E',
          2: '#4C8C72',
          3: '#7A9E8A',
          4: '#8C7A55',
          5: '#6B8790',
          6: '#9A7560',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
