const base = require('./jest.config.js');

module.exports = {
  ...base,
  setupFilesAfterEnv: ['<rootDir>/tests/setup-core.ts'],
};
