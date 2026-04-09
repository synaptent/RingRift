/**
 * Jest setup shared by non-React and supported-path suites.
 * This file intentionally avoids importing @testing-library/react so
 * rules/parity lanes can exit cleanly without React's MessagePort handle.
 */

process.env.ORCHESTRATOR_ADAPTER_ENABLED = 'true';

const RATE_LIMIT_ENV_KEYS = [
  'RATE_LIMIT_API_POINTS',
  'RATE_LIMIT_API_DURATION',
  'RATE_LIMIT_API_AUTH_POINTS',
  'RATE_LIMIT_API_AUTH_DURATION',
  'RATE_LIMIT_AUTH_POINTS',
  'RATE_LIMIT_AUTH_DURATION',
  'RATE_LIMIT_AUTH_LOGIN_POINTS',
  'RATE_LIMIT_AUTH_LOGIN_DURATION',
  'RATE_LIMIT_AUTH_REGISTER_POINTS',
  'RATE_LIMIT_GAME_POINTS',
  'RATE_LIMIT_GAME_MOVES_POINTS',
  'RATE_LIMIT_WS_POINTS',
  'RATE_LIMIT_GAME_CREATE_USER_POINTS',
  'RATE_LIMIT_GAME_CREATE_IP_POINTS',
  'RATE_LIMIT_WEBSOCKET_POINTS',
];
RATE_LIMIT_ENV_KEYS.forEach((key) => {
  delete process.env[key];
});

import { TextEncoder, TextDecoder } from 'util';
global.TextEncoder = TextEncoder as any;
global.TextDecoder = TextDecoder as any;

if (!(global as any).setImmediate) {
  (global as any).setImmediate = (fn: (...args: any[]) => void, ...args: any[]) =>
    setTimeout(fn, 0, ...args);
}

if (typeof window !== 'undefined' && !(window as any).setImmediate) {
  (window as any).setImmediate = (fn: (...args: any[]) => void, ...args: any[]) =>
    setTimeout(fn, 0, ...args);
}

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

jest.setTimeout(10000);

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() {
    return [];
  }
  unobserve() {}
} as any;

global.ResizeObserver = class ResizeObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
} as any;

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

afterEach(() => {
  jest.clearAllMocks();
});
