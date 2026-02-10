import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // REST
      '/chat': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // WebSocket
      '/ws': {
        target: 'ws://localhost:8001',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
