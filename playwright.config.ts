import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for RingRift end-to-end tests.
 *
 * These tests assume:
 * - PostgreSQL + Redis are running locally (see QUICKSTART.md).
 * - The Node backend and Vite client can be started via "npm run dev".
 *
 * The Playwright webServer config below will start "npm run dev" on
 * http://localhost:5173 and reuse an existing dev server when possible.
 */
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  fullyParallel: false,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'html' : 'list',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    video: 'off',
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
