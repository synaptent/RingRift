/**
 * Jest Setup File
 * Runs AFTER test framework is installed
 */

// LEGACY: Orchestrator adapter is now permanently enabled (2025-12-01).
// This env var assignment is kept for any old tests that might still check it,
// but the config no longer reads from environment variables - it's hardcoded to true.
// This line can be removed in a future cleanup pass (P20.7-2+).
process.env.ORCHESTRATOR_ADAPTER_ENABLED = 'true';

// Import Testing Library jest-dom matchers
import '@testing-library/jest-dom';

// Polyfill TextEncoder/TextDecoder for jsdom
import { TextEncoder, TextDecoder } from 'util';
global.TextEncoder = TextEncoder as any;
global.TextDecoder = TextDecoder as any;

// Polyfill setImmediate for environments (like jsdom) where it is not defined.
// Some dependencies used in server-side code (e.g. winston/readable-stream)
// assume setImmediate exists and will throw ReferenceError otherwise.
if (!(global as any).setImmediate) {
  (global as any).setImmediate = (fn: (...args: any[]) => void, ...args: any[]) =>
    setTimeout(fn, 0, ...args);
}

if (typeof window !== 'undefined' && !(window as any).setImmediate) {
  (window as any).setImmediate = (fn: (...args: any[]) => void, ...args: any[]) =>
    setTimeout(fn, 0, ...args);
}

// Mock import.meta for Vite-specific code
Object.defineProperty(global, 'import.meta', {
  value: {
    env: {
      MODE: 'test',
      DEV: false,
      PROD: false,
      SSR: false,
    },
  },
  writable: true,
});

// Global test timeout
jest.setTimeout(10000);

// Mock window.matchMedia for components using media queries
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // Deprecated
    removeListener: jest.fn(), // Deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() {
    return [];
  }
  unobserve() {}
} as any;

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
} as any;

// Mock localStorage (jsdom provides it, but add fallback)
if (!global.localStorage) {
  global.localStorage = {
    getItem: jest.fn(() => null),
    setItem: jest.fn(),
    removeItem: jest.fn(),
    clear: jest.fn(),
    length: 0,
    key: jest.fn(() => null),
  } as Storage;
}

// Mock console methods to reduce noise (optional)
// Uncomment if you want to suppress console output during tests
// global.console = {
//   ...console,
//   log: jest.fn(),
//   debug: jest.fn(),
//   info: jest.fn(),
//   warn: jest.fn(),
//   error: jest.fn(),
// };

// Global test cleanup
afterEach(() => {
  // Clear all mocks after each test
  jest.clearAllMocks();
});
