import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright Configuration for RingRift E2E Tests
 *
 * This configuration supports:
 * - Configurable timeouts (longer in CI for stability)
 * - CI-specific settings (retries, reporters, screenshots)
 * - Parallel/sequential execution control
 * - Environment-based base URL configuration
 *
 * Environment Variables:
 * - CI: Set to 'true' in CI environments (auto-set by GitHub Actions)
 * - E2E_BASE_URL: Override the base URL (default: http://localhost:5173)
 * - PLAYWRIGHT_WORKERS: Override worker count (default: 1 in CI, auto locally)
 *
 * Prerequisites:
 * - PostgreSQL + Redis running locally (see QUICKSTART.md)
 * - The Node backend and Vite client can be started via "npm run dev"
 *
 * Usage:
 * - Local development: npm run test:e2e
 * - Interactive mode:  npm run test:e2e:ui
 * - Headed browser:    npm run test:e2e:headed
 * - View report:       npm run test:e2e:report
 * - CI mode:           CI=true npm run test:e2e
 */

// CI detection
const CI = process.env.CI === 'true';

// Configuration constants
const BASE_URL = process.env.E2E_BASE_URL || 'http://localhost:5173';
const WORKERS = process.env.PLAYWRIGHT_WORKERS
  ? parseInt(process.env.PLAYWRIGHT_WORKERS, 10)
  : CI
    ? 1 // Single worker in CI for stability and predictable resource usage
    : undefined; // Auto-detect locally based on CPU cores

export default defineConfig({
  testDir: './tests/e2e',

  /* ============================================
   * Snapshot Configuration for Visual Regression Testing
   * ============================================ */
  snapshotDir: './tests/e2e/__snapshots__',
  snapshotPathTemplate:
    '{snapshotDir}/{testFileDir}/{testFileName}-snapshots/{arg}{-projectName}{ext}',

  /* ============================================
   * Timeout Configuration
   * CI environments get longer timeouts for stability
   * ============================================ */
  timeout: CI ? 60_000 : 30_000, // Per-test timeout: 60s CI, 30s local
  expect: {
    timeout: CI ? 10_000 : 5_000, // Assertion timeout: 10s CI, 5s local
    /* ============================================
     * Visual Regression Comparison Settings
     * ============================================ */
    toHaveScreenshot: {
      maxDiffPixels: 100, // Allow minor differences (anti-aliasing, font rendering)
      threshold: 0.2, // 20% color threshold per pixel
      animations: 'disabled', // Disable CSS animations for consistent screenshots
    },
    toMatchSnapshot: {
      maxDiffPixelRatio: 0.01, // Allow 1% pixel difference
    },
  },

  /* ============================================
   * Execution Configuration
   * ============================================ */
  fullyParallel: false, // Tests run sequentially within files
  workers: WORKERS,

  /* ============================================
   * Retry Configuration
   * More retries in CI to handle flaky infrastructure
   * ============================================ */
  retries: CI ? 2 : 0,

  /* ============================================
   * Reporter Configuration
   * CI: GitHub annotations + HTML report for artifacts
   * Local: Simple list output for fast feedback
   * ============================================ */
  reporter: CI
    ? [
        ['github'], // GitHub Actions annotations
        ['html', { open: 'never' }], // HTML report (don't auto-open in CI)
      ]
    : 'list',

  /* ============================================
   * Browser Configuration
   * ============================================ */
  use: {
    baseURL: BASE_URL,

    // Tracing - capture on first retry to debug failures
    trace: CI ? 'on-first-retry' : 'off',

    // Screenshots - capture on failure in CI for debugging
    screenshot: CI ? 'only-on-failure' : 'off',

    // Video - disabled by default (enable for specific debugging)
    video: 'off',

    // Action timeout for click, fill, etc.
    actionTimeout: CI ? 15_000 : 10_000,

    // Navigation timeout
    navigationTimeout: CI ? 30_000 : 15_000,
  },

  /* ============================================
   * Web Server Configuration
   * Starts the dev server before running tests
   * ============================================ */
  webServer: {
    command: 'npm run dev',
    url: BASE_URL,
    timeout: 120_000, // 2 minutes to start (includes build time)
    reuseExistingServer: !CI, // Reuse running dev server locally
    stdout: 'pipe', // Capture stdout for debugging
    stderr: 'pipe', // Capture stderr for debugging
  },

  /* ============================================
   * Project Configuration
   * Multi-browser testing support:
   * - Desktop: Chromium, Firefox, WebKit
   * - Mobile: Pixel 5 (Chrome), iPhone 12 (Safari)
   *
   * Run specific browsers:
   *   npm run test:e2e:chromium
   *   npm run test:e2e:firefox
   *   npm run test:e2e:webkit
   *   npm run test:e2e:all
   *
   * Install all browsers:
   *   npx playwright install
   *   npx playwright install firefox webkit (for specific browsers)
   *
   * Browser-specific notes:
   * - WebKit may have WebSocket timing differences
   * - Mobile viewports useful for responsive testing
   * ============================================ */
  projects: [
    // Desktop browsers
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    // Mobile viewports (valuable for responsive testing)
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },
  ],

  /* ============================================
   * Output Configuration
   * ============================================ */
  outputDir: 'test-results/', // Test artifacts directory
});
