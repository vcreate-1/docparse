import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    // Same-origin /api in dev, so PDF.js range requests need no CORS setup.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
