/**
 * Jest Environment Setup
 * Runs BEFORE test framework is installed
 * Used for mocking global browser APIs in Node environment
 */

// Mock localStorage for Node environment
const localStorageMock = {
  getItem: (key: string) => null,
  setItem: (key: string, value: string) => {},
  removeItem: (key: string) => {},
  clear: () => {},
  length: 0,
  key: (index: number) => null,
};

// Define localStorage on global object
Object.defineProperty(global, 'localStorage', {
  value: localStorageMock,
  writable: true,
  configurable: true,
});

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error';

// Enable FSM validation in active mode for Jest tests by default, unless
// explicitly overridden by the environment. This ensures that FSM validation
// is authoritative in tests while leaving runtime defaults unchanged.
if (!process.env.RINGRIFT_FSM_VALIDATION_MODE) {
  process.env.RINGRIFT_FSM_VALIDATION_MODE = 'active';
}
