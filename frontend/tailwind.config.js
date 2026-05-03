/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#ffffff',
          hover: '#f8f9fa',
          active: '#f1f3f5',
        },
        'border-subtle': '#e5e7eb',
        'border-hover': '#d1d5db',
        accent: {
          DEFAULT: '#2563eb',
          dim: 'rgba(37,99,235,0.08)',
          strong: '#1d4ed8',
        },
        'text-primary': '#111827',
        'text-secondary': '#6b7280',
        'text-muted': '#9ca3af',
        success: '#16a34a',
        warning: '#d97706',
        danger: '#dc2626',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
      boxShadow: {
        glow: '0 4px 14px rgba(37,99,235,0.15)',
        'glow-sm': '0 2px 8px rgba(37,99,235,0.1)',
      },
    },
  },
  plugins: [],
}
