import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import { visualizer } from 'rollup-plugin-visualizer';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import path from 'path';

/**
 * Allowlist of env vars that are safe to bake into the public client bundle.
 *
 * Anything returned here ends up in plain text in every browser's JS, so
 * additions must be explicitly reviewed. See the SECURITY note on the
 * `define['process.env']` site below.
 */
const PUBLIC_CLIENT_ENV_KEYS = ['NODE_ENV', 'RINGRIFT_AI_SERVICE_URL'] as const;

/**
 * Patterns that should never appear in a value we're about to inline into the
 * public bundle. If any allowlisted env var unexpectedly contains a
 * credential-shaped value (because of a misconfigured deploy), fail the build
 * loudly instead of shipping it.
 */
const SECRET_SHAPED_PATTERNS: Array<{ name: string; re: RegExp }> = [
  { name: 'Slack webhook', re: /hooks\.slack\.com\/services\/[A-Z0-9/]+/ },
  { name: 'AWS access key id', re: /\bAKIA[0-9A-Z]{16}\b/ },
  {
    name: 'AWS secret access key shape',
    re: /(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])/,
  },
  { name: 'postgres URL', re: /\bpostgres(?:ql)?:\/\/[^\s"']+/ },
  { name: 'JWT-shaped 64+ hex token', re: /\b[a-f0-9]{64,}\b/ },
  { name: 'GitHub token', re: /\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}/ },
  { name: 'private key block', re: /-----BEGIN [A-Z ]*PRIVATE KEY-----/ },
];

function buildClientProcessEnv(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const key of PUBLIC_CLIENT_ENV_KEYS) {
    const raw = process.env[key];
    if (typeof raw !== 'string' || raw.length === 0) continue;
    for (const { name, re } of SECRET_SHAPED_PATTERNS) {
      if (re.test(raw)) {
        throw new Error(
          `[vite.config] Refusing to bake env var ${key} into the public client ` +
            `bundle: value matches ${name} pattern. If this is intentional, ` +
            `prove it's safe and update SECRET_SHAPED_PATTERNS.`
        );
      }
    }
    out[key] = raw;
  }
  return out;
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    // PWA support for offline sandbox play
    VitePWA({
      registerType: 'autoUpdate',
      // The generated manifest is normally build-only. Required browser tests
      // exercise the dev server, so expose the same plugin-owned asset in CI.
      devOptions: { enabled: process.env.CI === 'true' },
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
    // SECURITY: build-time substitutions baked into the public client bundle.
    //
    // Anything written here is shipped to every browser that loads the site
    // and is permanently cached at its hashed URL. Treat this surface as
    // strictly public.
    //
    // History: a prior `startsWith('RINGRIFT_')` wildcard exposed
    // RINGRIFT_SLACK_WEBHOOK on the public CDN because src/shared/* code
    // references process.env for debug flags (CanonicalReplayEngine,
    // engine/core, engine/aggregates), which caused Vite to inline the
    // whole filtered process.env object into the client bundle. The
    // minifier then preserved every key because of dynamic access in
    // envFlags.ts. See commit 435f81e50 for the regression.
    //
    // Rule: explicit allowlist only. Add a new key here only if (a) the
    // browser genuinely needs it and (b) the value is safe to make public.
    'process.env': buildClientProcessEnv(),
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
