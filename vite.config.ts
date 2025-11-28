import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // Bundle analyzer - generates stats.html in dist folder
    visualizer({
      filename: 'dist/client/stats.html',
      open: false,
      gzipSize: true,
      brotliSize: true,
    }),
  ],
  root: 'src/client',
  build: {
    outDir: '../../dist/client',
    emptyOutDir: true,
    // Enable chunk size warnings at 500KB
    chunkSizeWarningLimit: 500,
    rollupOptions: {
      output: {
        // Manual chunks for better code splitting
        manualChunks: {
          // Core React vendor chunk
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          // UI libraries chunk
          'vendor-ui': ['clsx', 'tailwind-merge', 'react-hot-toast'],
          // Socket.io chunk for real-time features
          'vendor-socket': ['socket.io-client'],
          // Data fetching and state management
          'vendor-query': ['@tanstack/react-query', 'axios'],
        },
      },
    },
  },
  server: {
    // Use the standard Vite dev port (5173) to match .env.example, CORS_ORIGIN,
    // and documentation. The backend API + WebSockets run on 3000 in dev.
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
      },
      '/socket.io': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  define: {
    'process.env': Object.fromEntries(
      Object.entries(process.env).filter(
        ([key, value]) =>
          (key === 'NODE_ENV' || key.startsWith('RINGRIFT_')) && typeof value === 'string'
      )
    ),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/client': path.resolve(__dirname, './src/client'),
      '@/shared': path.resolve(__dirname, './src/shared'),
    },
  },
});
