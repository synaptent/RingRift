const base = require('./jest.supported-path.config.js');

module.exports = {
  ...base,
  collectCoverage: true,
  coverageDirectory: 'coverage/rules-critical',
  coverageReporters: ['text-summary', 'json-summary', 'lcov'],
  collectCoverageFrom: [
    'src/shared/engine/**/*.ts',
    '!src/shared/engine/**/*.d.ts',
    '!src/shared/engine/index.ts',
    '!src/shared/engine/**/index.ts',
    '!src/shared/engine/legacy/**',
    '!src/shared/engine/contracts/testVectorGenerator.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 45,
      functions: 59,
      lines: 60,
      statements: 59,
    },
  },
};
