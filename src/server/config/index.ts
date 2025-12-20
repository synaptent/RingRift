/**
 * Configuration Module - Canonical Entry Point
 *
 * This is the single canonical entry point for all application configuration.
 * All server code should import from this module:
 *
 * Usage:
 *   import { config } from './config';
 *   // or
 *   import { config, validateSecretsOrThrow } from './config';
 *   // or for topology enforcement
 *   import { config, enforceAppTopology } from './config';
 *
 * Architecture:
 * - `env.ts` - Raw environment variable schema definitions
 * - `unified.ts` - Config assembly and validation logic
 * - `topology.ts` - Deployment topology enforcement
 * - `index.ts` (this file) - Canonical re-export point
 */

// ============================================================================
// Primary Configuration Export
// ============================================================================

// Export the main configuration object from unified module
export { config } from './unified';
export type { AppConfig } from './unified';

// ============================================================================
// Topology Enforcement
// ============================================================================

export { enforceAppTopology } from './topology';

// ============================================================================
// Environment Schema & Utilities
// ============================================================================

export {
  EnvSchema,
  NodeEnvSchema,
  AppTopologySchema,
  RulesModeSchema,
  LogLevelSchema,
  LogFormatSchema,
  parseEnv,
  loadEnvOrExit,
  getEffectiveNodeEnv,
  isProduction,
  isStaging,
  isDevelopment,
  isTest,
  isProductionLike,
  parseDurationToSeconds,
} from './env';

export type {
  RawEnv,
  EnvValidationResult,
  NodeEnv,
  AppTopology,
  RulesMode,
  LogLevel,
  LogFormat,
} from './env';

// ============================================================================
// Secrets Validation Utilities
// ============================================================================

export {
  validateSecretsOrThrow,
  validateAllSecrets,
  validateSecret,
  isPlaceholderSecret,
  getSecretsDocumentation,
  SECRET_DEFINITIONS,
  SECRET_MIN_LENGTHS,
  PLACEHOLDER_SECRETS,
} from '../utils/secretsValidation';

export type { SecretDefinition, SecretValidationResult } from '../utils/secretsValidation';
