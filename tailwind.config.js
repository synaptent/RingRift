/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/client/**/*.{js,ts,jsx,tsx}",
    "./src/client/index.html",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        secondary: {
          50: '#f8fafc',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      // Animation keyframes for game UX
      keyframes: {
        // Smooth position transition for moves
        'piece-move': {
          '0%': { transform: 'translate(var(--from-x), var(--from-y))' },
          '100%': { transform: 'translate(0, 0)' }
        },
        // Pulse for selection
        'selection-pulse': {
          '0%, 100%': { boxShadow: '0 0 0 2px rgba(var(--ring-color), 0.4)' },
          '50%': { boxShadow: '0 0 0 4px rgba(var(--ring-color), 0.6)' }
        },
        // Bounce for captures
        'capture-bounce': {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.15)' }
        },
        // Fade in for new pieces
        'piece-appear': {
          '0%': { opacity: '0', transform: 'scale(0.8)' },
          '100%': { opacity: '1', transform: 'scale(1)' }
        },
        // Victory celebration
        'celebrate': {
          '0%': { transform: 'scale(1) rotate(0deg)' },
          '25%': { transform: 'scale(1.1) rotate(-5deg)' },
          '75%': { transform: 'scale(1.1) rotate(5deg)' },
          '100%': { transform: 'scale(1) rotate(0deg)' }
        }
      },
      // Animation utility classes
      animation: {
        'piece-move': 'piece-move 300ms ease-out',
        'selection-pulse': 'selection-pulse 1.5s ease-in-out infinite',
        'capture-bounce': 'capture-bounce 400ms ease-out',
        'piece-appear': 'piece-appear 200ms ease-out',
        'celebrate': 'celebrate 500ms ease-in-out'
      },
      // Transition property utilities
      transitionProperty: {
        'position': 'left, top, transform',
        'colors-shadow': 'color, background-color, border-color, box-shadow'
      },
      // Transition duration utilities
      transitionDuration: {
        '250': '250ms',
        '350': '350ms'
      },
      // Transition timing function utilities
      transitionTimingFunction: {
        'bounce-out': 'cubic-bezier(0.34, 1.56, 0.64, 1)'
      }
    },
  },
  plugins: [],
}