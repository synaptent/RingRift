import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import { visualizer } from 'rollup-plugin-visualizer';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    // PWA support for offline sandbox play
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'ringrift-icon.png', 'apple-touch-icon.png'],
      manifest: {
        name: 'RingRift - Multiplayer Strategy Game',
        short_name: 'RingRift',
        description: 'Place rings, form lines, and claim territory on dynamic board geometries.',
        theme_color: '#0f172a',
        background_color: '#0f172a',
        display: 'standalone',
        scope: '/',
        start_url: '/sandbox',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png',
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable',
          },
        ],
      },
      workbox: {
        // Cache game engine JS/CSS for offline sandbox play
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        // Skip socket.io and API routes - they require network
        navigateFallback: 'index.html',
        navigateFallbackDenylist: [/^\/api\//, /^\/socket\.io\//],
        runtimeCaching: [
          {
            // Cache Bunny Fonts for offline
            urlPattern: /^https:\/\/fonts\.bunny\.net\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'bunny-fonts-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365, // 1 year
              },
              cacheableResponse: {
                statuses: [0, 200],
              },
            },
          },
        ],
      },
    }),
    // Bundle analyzer - generates stats.html in dist folder
    visualizer({
      filename: 'dist/client/stats.html',
      open: false,
      gzipSize: true,
      brotliSize: true,
    }),
    // Copy contract test vectors into the built client so the sandbox scenario
    // browser can load them in both local and production builds.
    viteStaticCopy({
      targets: [
        {
          src: '../../tests/fixtures/contract-vectors/v2/*.vectors.json',
          dest: 'scenarios/vectors',
        },
      ],
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
    // Expose NODE_ENV and all RINGRIFT_* environment variables to the client bundle.
    // This includes RINGRIFT_AI_SERVICE_URL for the replay service.
    'process.env': Object.fromEntries(
      Object.entries(process.env).filter(
        ([key, value]) =>
          (key === 'NODE_ENV' || key.startsWith('RINGRIFT_')) && typeof value === 'string'
      )
    ),
    // Inject Vite env to a global for Jest compatibility (avoids import.meta parse errors)
    'globalThis.__VITE_ENV__': 'import.meta.env',
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/client': path.resolve(__dirname, './src/client'),
      '@/shared': path.resolve(__dirname, './src/shared'),
    },
  },
}));
