/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0B0F19', // Deep gunmetal/slate
        surface: '#151C2C',    // Lighter slate for cards
        surfaceHover: '#1E293B',
        primary: '#A3E635',    // Neon Lime Green (Success/Action)
        warning: '#F97316',    // Signal Orange
        danger: '#EF4444',     // Red
        textMain: '#F8FAFC',   // Slate 50
        textMuted: '#94A3B8',  // Slate 400
        borderColor: '#334155' // Slate 700
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['"Space Grotesk"', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
