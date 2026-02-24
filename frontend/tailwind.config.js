/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'ci-primary': '#2E86C1',
        'ci-success': '#27AE60',
        'ci-danger': '#E74C3C',
        'ci-warning': '#F39C12',
        'ci-purple': '#8E44AD',
        'ci-teal': '#1ABC9C',
        'ci-gray': '#95A5A6',
        'ci-bg': '#F0F2F6',
        'ci-text': '#1B2A4A',
        'ci-dark-bg': '#0f172a',
        'ci-dark-card': '#1e293b',
        'ci-dark-text': '#e2e8f0',
      },
    },
  },
  plugins: [],
}
