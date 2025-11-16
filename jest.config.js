/** @type {import('jest').Config} */
module.exports = {
  // Use ts-jest preset for TypeScript support
  preset: 'ts-jest',
  
  // Run tests in custom Node environment (with localStorage mock)
  testEnvironment: '<rootDir>/tests/test-environment.js',
  
  // Roots for test discovery
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  
  // Test match patterns
  testMatch: [
    '**/__tests__/**/*.+(ts|tsx|js)',
    '**/?(*.)+(spec|test).+(ts|tsx|js)'
  ],
  
  // Module paths
  modulePaths: ['<rootDir>/src'],
  
  // Transform files with ts-jest
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
  },
  
  // Coverage configuration
  collectCoverage: false, // Enable with --coverage flag
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/index.ts',
    '!src/client/index.tsx',
    '!src/server/index.ts',
    '!src/**/__tests__/**',
    '!src/**/*.test.{ts,tsx}',
    '!src/**/*.spec.{ts,tsx}',
  ],
  
  // Coverage thresholds (80% target)
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  
  // Coverage reporters
  coverageReporters: [
    'text',
    'text-summary',
    'html',
    'lcov',
    'json',
  ],
  
  // Setup files - runs AFTER test framework is installed
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  
  
  // Module name mapper for path aliases
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@shared/(.*)$': '<rootDir>/src/shared/$1',
    '^@server/(.*)$': '<rootDir>/src/server/$1',
    '^@client/(.*)$': '<rootDir>/src/client/$1',
  },
  
  // Ignore patterns
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
  ],
  
  // Global test timeout (10 seconds)
  testTimeout: 10000,
  
  // Verbose output
  verbose: true,
  
  // Clear mocks between tests
  clearMocks: true,
  
  // Restore mocks between tests
  restoreMocks: true,
  
  // Reset mocks between tests
  resetMocks: true,
};
