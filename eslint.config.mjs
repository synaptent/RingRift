import tseslint from '@typescript-eslint/eslint-plugin';
import tsParser from '@typescript-eslint/parser';
import reactPlugin from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import unicorn from 'eslint-plugin-unicorn';
import jestPlugin from 'eslint-plugin-jest';
import globals from 'globals';

// Flat config for ESLint 9+, migrated from .eslintrc.json while preserving
// the existing rule set and TypeScript/React integration.

export default [
  // Global ignores (was `ignorePatterns` in .eslintrc.json)
  {
    ignores: [
      'dist/**',
      'node_modules/**',
      'coverage/**',
      '*.config.js',
      '*.config.ts',
      '.eslintrc.json',
      'tests/test-environment.js',
    ],
  },

  // Base config: rely on the project-specific rules below rather than
  // spreading upstream presets, to keep behaviour close to the previous
  // .eslintrc.json while avoiding flat-config shape issues.

  // Project-specific TypeScript/React rules
  {
    files: ['src/**/*.{ts,tsx,js,jsx}', 'tests/**/*.{ts,tsx,js,jsx}', 'scripts/**/*.ts'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        ecmaFeatures: { jsx: true },
        project: ['./tsconfig.eslint.json'],
      },
      globals: {
        ...globals.browser,
        ...globals.node,
        ...globals.es2021,
      },
    },
    plugins: {
      '@typescript-eslint': tseslint,
      react: reactPlugin,
      'react-hooks': reactHooks,
      unicorn,
    },
    settings: {
      react: {
        version: 'detect',
      },
    },
    rules: {
      // From .eslintrc.json
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      '@typescript-eslint/explicit-module-boundary-types': 'off',
      '@typescript-eslint/no-non-null-assertion': 'warn',
      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off',
      // Stricter console rule: error to block commits with console.log
      // Allow: debug (for guarded diagnostics), warn, error
      'no-console': ['error', { allow: ['debug', 'warn', 'error'] }],
      'no-constant-condition': ['error', { checkLoops: false }],
      'no-case-declarations': 'off',
      'prefer-const': 'warn',
    },
  },

  // Debug utility override - allow console statements in debug tools
  {
    files: ['src/client/utils/freezeDebugger.ts'],
    rules: {
      'no-console': 'off',
    },
  },

  // Jest/test overrides (was `overrides` in .eslintrc.json)
  {
    files: ['tests/**/*.{ts,tsx,js,jsx}'],
    languageOptions: {
      globals: {
        ...globals.jest,
      },
    },
    plugins: {
      jest: jestPlugin,
    },
    rules: {
      '@typescript-eslint/no-var-requires': 'off',
      // Prevent .only() calls from being committed - these cause CI to skip tests
      'jest/no-focused-tests': 'error',
      // Warn about .skip() calls - these should have justification comments
      // See docs/supplementary/TEST_SKIPPED_TRIAGE.md for categorization
      'jest/no-disabled-tests': 'warn',
    },
  },
];
