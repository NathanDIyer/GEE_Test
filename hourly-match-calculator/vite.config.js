import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/GEE_Test/hourly-match-calculator/',  // GitHub Pages base path
  server: {
    port: 3000,
  },
})
