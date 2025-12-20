#!/usr/bin/env ts-node
/**
 * Deployment Configuration Validation Script
 *
 * Validates that deployment configuration files (docker-compose.yml, docker-compose.staging.yml)
 * are consistent with the environment variable schema (.env.example) and follow best practices.
 *
 * Usage:
 *   npx ts-node scripts/validate-deployment-config.ts
 *   npm run validate:deployment
 */

import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Types
// ============================================================================

interface ValidationResult {
  file: string;
  valid: boolean;
  errors: string[];
  warnings: string[];
}

interface ServiceConfig {
  name: string;
  environment: Map<string, string>;
  volumes: string[];
  ports: string[];
  hasHealthcheck: boolean;
  hasResourceLimits: boolean;
  dependsOn: string[];
}

interface DockerComposeConfig {
  services: Map<string, ServiceConfig>;
  volumes: string[];
  networks: string[];
}

export interface DeploymentConfigValidationResult {
  ok: boolean;
  errors: string[];
  warnings: string[];
  results: ValidationResult[];
}

// ============================================================================
// Environment Variable Parser
// ============================================================================

/**
 * Parse .env.example to get all defined variable names and their example values.
 * Also parses commented-out variable definitions (e.g., # RINGRIFT_RULES_MODE=shadow)
 * as these are still documented variables.
 */
function parseEnvExample(filePath: string): Map<string, string> {
  const envVars = new Map<string, string>();

  if (!fs.existsSync(filePath)) {
    console.error(`❌ Error: ${filePath} not found`);
    process.exit(1);
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === '') continue;

    // Parse VAR=value or VAR= (empty value)
    const match = trimmed.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
    if (match) {
      const [, name, value] = match;
      envVars.set(name, value);
      continue;
    }

    // Also parse commented variables (e.g., # RINGRIFT_RULES_MODE=shadow)
    // These are documented options even if not active by default
    const commentedMatch = trimmed.match(/^#\s*([A-Z_][A-Z0-9_]*)=(.*)$/);
    if (commentedMatch) {
      const [, name, value] = commentedMatch;
      // Only add if not already defined (uncommented takes precedence)
      if (!envVars.has(name)) {
        envVars.set(name, value);
      }
    }
  }

  return envVars;
}

// ============================================================================
// Docker Compose Parser (Simple YAML-like parser)
// ============================================================================

/**
 * Simple docker-compose YAML parser
 * This handles the specific structure we use without requiring js-yaml
 */
function parseDockerCompose(filePath: string): DockerComposeConfig | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  const config: DockerComposeConfig = {
    services: new Map(),
    volumes: [],
    networks: [],
  };

  let currentSection = '';
  let currentService = '';
  let currentServiceSection = '';

  for (const line of lines) {
    // Skip empty lines and comments
    if (line.trim() === '' || line.trim().startsWith('#')) continue;

    // Count leading spaces
    const leadingSpaces = line.match(/^(\s*)/)?.[1].length || 0;
    const trimmed = line.trim();

    // Top-level sections (services:, volumes:, networks:)
    if (leadingSpaces === 0 && trimmed.endsWith(':')) {
      currentSection = trimmed.slice(0, -1);
      currentService = '';
      currentServiceSection = '';
      continue;
    }

    // Service names (2-space indent under services:)
    if (currentSection === 'services' && leadingSpaces === 2 && trimmed.endsWith(':')) {
      currentService = trimmed.slice(0, -1);
      currentServiceSection = '';

      if (!config.services.has(currentService)) {
        config.services.set(currentService, {
          name: currentService,
          environment: new Map(),
          volumes: [],
          ports: [],
          hasHealthcheck: false,
          hasResourceLimits: false,
          dependsOn: [],
        });
      }
      continue;
    }

    // Service subsections (4-space indent)
    if (
      currentSection === 'services' &&
      currentService &&
      leadingSpaces === 4 &&
      trimmed.endsWith(':')
    ) {
      currentServiceSection = trimmed.slice(0, -1);
      continue;
    }

    // Parse service content
    if (currentSection === 'services' && currentService) {
      const service = config.services.get(currentService)!;

      // Environment variables
      if (currentServiceSection === 'environment' && leadingSpaces >= 6) {
        // Format: - VAR=value or - VAR=${...}
        const envMatch = trimmed.match(/^-\s*([A-Z_][A-Z0-9_]*)=(.*)$/);
        if (envMatch) {
          service.environment.set(envMatch[1], envMatch[2]);
        }
      }

      // Volumes
      if (currentServiceSection === 'volumes' && leadingSpaces >= 6 && trimmed.startsWith('-')) {
        service.volumes.push(trimmed.slice(1).trim());
      }

      // Ports
      if (currentServiceSection === 'ports' && leadingSpaces >= 6 && trimmed.startsWith('-')) {
        service.ports.push(trimmed.slice(1).trim().replace(/'/g, ''));
      }

      // depends_on
      if (currentServiceSection === 'depends_on' && leadingSpaces >= 6) {
        if (trimmed.startsWith('-')) {
          service.dependsOn.push(trimmed.slice(1).trim());
        } else if (trimmed.endsWith(':')) {
          // Service with condition
          service.dependsOn.push(trimmed.slice(0, -1));
        }
      }

      // Healthcheck detection
      if (trimmed === 'healthcheck:' || currentServiceSection === 'healthcheck') {
        service.hasHealthcheck = true;
      }

      // Resource limits detection (under deploy.resources.limits)
      if (trimmed.includes('memory:') && leadingSpaces >= 8) {
        service.hasResourceLimits = true;
      }
    }

    // Volume definitions
    if (currentSection === 'volumes' && leadingSpaces === 2 && trimmed.endsWith(':')) {
      config.volumes.push(trimmed.slice(0, -1));
    }

    // Network definitions
    if (currentSection === 'networks' && leadingSpaces === 2 && trimmed.endsWith(':')) {
      config.networks.push(trimmed.slice(0, -1));
    }
  }

  return config;
}

// ============================================================================
// Hardcoded Secret Detection
// ============================================================================

const SENSITIVE_VAR_PATTERNS = [
  /SECRET/i,
  /PASSWORD/i,
  /TOKEN/i,
  /KEY/i,
  /CREDENTIAL/i,
  /API_KEY/i,
];

const PLACEHOLDER_PATTERNS = [
  /change.*this/i,
  /your.*secret/i,
  /changeme/i,
  /placeholder/i,
  /example/i,
  /password$/i,
];

/**
 * Check if a variable name is sensitive
 */
function isSensitiveVar(name: string): boolean {
  return SENSITIVE_VAR_PATTERNS.some((pattern) => pattern.test(name));
}

/**
 * Check if a value looks like a hardcoded secret (not a variable reference)
 */
function isHardcodedSecret(name: string, value: string): boolean {
  if (!isSensitiveVar(name)) return false;

  // Variable references are not hardcoded
  if (value.includes('${')) return false;

  // Empty values are not hardcoded secrets
  if (value === '' || value === '""' || value === "''") return false;

  // Check for placeholder patterns
  if (PLACEHOLDER_PATTERNS.some((pattern) => pattern.test(value))) {
    return true;
  }

  // Non-empty literal values for sensitive vars are suspicious
  return value.length > 0 && !value.startsWith('$');
}

// ============================================================================
// Validation Rules
// ============================================================================

/**
 * Validate a docker-compose file against .env.example and best practices
 */
function validateDockerCompose(
  composePath: string,
  composeConfig: DockerComposeConfig,
  envExampleVars: Map<string, string>,
  isStaging: boolean
): ValidationResult {
  const result: ValidationResult = {
    file: composePath,
    valid: true,
    errors: [],
    warnings: [],
  };

  const allReferencedVars = new Set<string>();

  for (const [serviceName, service] of composeConfig.services) {
    // Collect all referenced env vars
    for (const [varName, value] of service.environment) {
      allReferencedVars.add(varName);

      // Extract variables from ${VAR:-default} patterns
      const varRefs = value.matchAll(/\$\{([A-Z_][A-Z0-9_]*)/g);
      for (const match of varRefs) {
        allReferencedVars.add(match[1]);
      }

      // Check for hardcoded secrets
      if (isHardcodedSecret(varName, value)) {
        result.warnings.push(
          `[${serviceName}] Hardcoded secret detected: ${varName}=${value.slice(0, 20)}...`
        );
      }
    }

    // Check for health checks on main services (not optional monitoring)
    const mainServices = ['app', 'postgres', 'redis', 'ai-service'];
    if (mainServices.includes(serviceName)) {
      // Only flag missing health check in staging file (it should add them)
      if (isStaging && !service.hasHealthcheck) {
        // Check if service has condition-based depends_on (implies healthcheck elsewhere)
        const hasCondition = service.dependsOn.length === 0;
        if (hasCondition) {
          result.warnings.push(
            `[${serviceName}] No healthcheck configured for staging environment`
          );
        }
      }
    }

    // Check for resource limits in staging/production
    if (isStaging && !service.hasResourceLimits && mainServices.includes(serviceName)) {
      // Resource limits should be in base compose or staging override
    }

    // Check for port conflicts
    const portMappings = new Map<string, string>();
    for (const port of service.ports) {
      const hostPort = port.split(':')[0];
      if (portMappings.has(hostPort)) {
        result.errors.push(
          `[${serviceName}] Port conflict: ${hostPort} already mapped to ${portMappings.get(hostPort)}`
        );
        result.valid = false;
      }
      portMappings.set(hostPort, serviceName);
    }
  }

  // Check for env vars referenced but not in .env.example
  // Skip common Docker/system vars
  const systemVars = new Set([
    'NODE_ENV',
    'PYTHON_ENV',
    'POSTGRES_DB',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'GF_SECURITY_ADMIN_PASSWORD',
    'GF_USERS_ALLOW_SIGN_UP',
    'GF_SERVER_ROOT_URL',
  ]);

  for (const varName of allReferencedVars) {
    if (systemVars.has(varName)) continue;

    // Check if it's a variable that might be aliased from .env.example
    const hasInEnvExample = envExampleVars.has(varName);
    const hasRelatedVar = Array.from(envExampleVars.keys()).some(
      (k) => k.includes(varName) || varName.includes(k.replace('_', ''))
    );

    if (!hasInEnvExample && !hasRelatedVar) {
      // Special handling for vars that map to different names
      const knownMappings: Record<string, string> = {
        DB_PASSWORD: 'DB_PASSWORD',
        GRAFANA_PASSWORD: 'GRAFANA_PASSWORD',
        GRAFANA_ROOT_URL: 'GRAFANA_ROOT_URL',
      };

      if (!knownMappings[varName] && !varName.startsWith('GF_')) {
        result.warnings.push(
          `Environment variable ${varName} used in docker-compose but not documented in .env.example`
        );
      }
    }
  }

  return result;
}

/**
 * Validate that all volumes referenced in services are defined
 */
function validateVolumes(composeConfig: DockerComposeConfig): ValidationResult {
  const result: ValidationResult = {
    file: 'volumes',
    valid: true,
    errors: [],
    warnings: [],
  };

  const definedVolumes = new Set(composeConfig.volumes);

  for (const [serviceName, service] of composeConfig.services) {
    for (const volume of service.volumes) {
      // Named volumes have format volume_name:/path
      const parts = volume.split(':');
      if (parts.length >= 2) {
        const volumeName = parts[0];
        // Skip host path mounts (start with . or /)
        if (!volumeName.startsWith('.') && !volumeName.startsWith('/')) {
          if (!definedVolumes.has(volumeName)) {
            result.errors.push(`[${serviceName}] Undefined volume: ${volumeName}`);
            result.valid = false;
          }
        }
      }
    }
  }

  return result;
}

/**
 * Validate Dockerfile best practices
 */
function validateDockerfile(dockerfilePath: string): ValidationResult {
  const result: ValidationResult = {
    file: dockerfilePath,
    valid: true,
    errors: [],
    warnings: [],
  };

  if (!fs.existsSync(dockerfilePath)) {
    result.warnings.push('Dockerfile not found');
    return result;
  }

  const content = fs.readFileSync(dockerfilePath, 'utf-8');

  // Check for HEALTHCHECK
  if (!content.includes('HEALTHCHECK')) {
    result.warnings.push('Dockerfile does not include HEALTHCHECK instruction');
  }

  // Check for non-root user
  if (!content.includes('USER ') || content.includes('USER root')) {
    result.warnings.push('Dockerfile should run as non-root user');
  }

  // Check for multi-stage build
  if (!content.includes('AS builder') && !content.includes('AS runtime')) {
    result.warnings.push('Consider using multi-stage build for smaller images');
  }

  // Check for .dockerignore reference
  const dockerignorePath = path.join(path.dirname(dockerfilePath), '.dockerignore');
  if (!fs.existsSync(dockerignorePath)) {
    result.warnings.push(
      '.dockerignore file not found - consider adding one to reduce build context'
    );
  }

  return result;
}

/**
 * Cross-validate env.ts schema against .env.example
 */
function validateEnvSchema(
  envExampleVars: Map<string, string>,
  envSchemaPath: string
): ValidationResult {
  const result: ValidationResult = {
    file: envSchemaPath,
    valid: true,
    errors: [],
    warnings: [],
  };

  if (!fs.existsSync(envSchemaPath)) {
    result.warnings.push('env.ts schema file not found');
    return result;
  }

  const content = fs.readFileSync(envSchemaPath, 'utf-8');

  // Extract variable names from the schema
  const schemaVars = new Set<string>();
  const varMatches = content.matchAll(/^\s*([A-Z_][A-Z0-9_]*):/gm);
  for (const match of varMatches) {
    schemaVars.add(match[1]);
  }

  // Check for vars in .env.example but not in schema
  for (const [varName] of envExampleVars) {
    // Skip VITE_ vars (client-side only)
    if (varName.startsWith('VITE_')) continue;
    // Skip SMTP vars (optional)
    if (varName.startsWith('SMTP_')) continue;
    // Skip individual DB component vars (used by some tools)
    if (['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'].includes(varName)) continue;
    // Skip REDIS component vars
    if (['REDIS_HOST', 'REDIS_PORT'].includes(varName)) continue;
    // Skip SOCKET_PORT (legacy)
    if (varName === 'SOCKET_PORT') continue;
    // Skip AI_LOG_LEVEL and AI_SERVICE_PORT (AI service specific)
    if (['AI_LOG_LEVEL', 'AI_SERVICE_PORT'].includes(varName)) continue;
    // Skip debug flags
    if (varName.startsWith('RINGRIFT_ENABLE_') || varName.startsWith('RINGRIFT_SANDBOX_')) continue;
    if (varName === 'RINGRIFT_LOCAL_AI_HEURISTIC_MODE') continue;

    if (!schemaVars.has(varName)) {
      result.warnings.push(`Variable ${varName} in .env.example but not in Zod schema`);
    }
  }

  return result;
}

/**
 * Validate .env.staging for correct Phase 1 orchestrator configuration
 */
function validateStagingEnv(stagingEnvPath: string): ValidationResult {
  const result: ValidationResult = {
    file: stagingEnvPath,
    valid: true,
    errors: [],
    warnings: [],
  };

  if (!fs.existsSync(stagingEnvPath)) {
    result.warnings.push('.env.staging not found - skipping staging config validation');
    return result;
  }

  const envVars = parseEnvExample(stagingEnvPath);

  // Phase 3+ Orchestrator Configuration Checks
  // Note: ORCHESTRATOR_ROLLOUT_PERCENTAGE and ORCHESTRATOR_SHADOW_MODE_ENABLED
  // were removed in Phase 3 - orchestrator is now permanently enabled at 100%
  const requiredConfig = {
    RINGRIFT_RULES_MODE: 'ts',
    ORCHESTRATOR_ADAPTER_ENABLED: 'true',
  };

  for (const [key, expectedValue] of Object.entries(requiredConfig)) {
    const actualValue = envVars.get(key);
    if (actualValue !== expectedValue) {
      result.errors.push(
        `Invalid Phase 1 configuration: ${key} should be "${expectedValue}", found "${actualValue || 'undefined'}"`
      );
      result.valid = false;
    }
  }

  return result;
}

/**
 * Validate CI workflow for required orchestrator jobs
 */
function validateCIWorkflow(ciWorkflowPath: string): ValidationResult {
  const result: ValidationResult = {
    file: ciWorkflowPath,
    valid: true,
    errors: [],
    warnings: [],
  };

  if (!fs.existsSync(ciWorkflowPath)) {
    result.errors.push('CI workflow file not found');
    result.valid = false;
    return result;
  }

  const content = fs.readFileSync(ciWorkflowPath, 'utf-8');

  const requiredJobs = [
    'orchestrator-soak-smoke',
    'orchestrator-short-soak',
    'orchestrator-parity',
  ];

  for (const job of requiredJobs) {
    if (!content.includes(`${job}:`)) {
      result.errors.push(`Missing required CI job: ${job}`);
      result.valid = false;
    }
  }

  return result;
}

// ============================================================================
// Main Validation
// ============================================================================

function validateDeploymentConfigs(): ValidationResult[] {
  const results: ValidationResult[] = [];
  const projectRoot = path.resolve(__dirname, '..');

  console.log('🔍 Validating deployment configurations...\n');

  // Parse .env.example
  const envExamplePath = path.join(projectRoot, '.env.example');
  console.log(`📄 Parsing ${envExamplePath}`);
  const envExampleVars = parseEnvExample(envExamplePath);
  console.log(`   Found ${envExampleVars.size} environment variables\n`);

  // Validate docker-compose.yml
  const dockerComposePath = path.join(projectRoot, 'docker-compose.yml');
  console.log(`📄 Validating ${dockerComposePath}`);
  const dockerComposeConfig = parseDockerCompose(dockerComposePath);
  if (dockerComposeConfig) {
    const composeResult = validateDockerCompose(
      dockerComposePath,
      dockerComposeConfig,
      envExampleVars,
      false
    );
    results.push(composeResult);

    // Validate volumes
    const volumeResult = validateVolumes(dockerComposeConfig);
    if (volumeResult.errors.length > 0 || volumeResult.warnings.length > 0) {
      results.push(volumeResult);
    }

    console.log(`   Found ${dockerComposeConfig.services.size} services\n`);
  } else {
    results.push({
      file: dockerComposePath,
      valid: false,
      errors: ['docker-compose.yml not found or could not be parsed'],
      warnings: [],
    });
  }

  // Validate docker-compose.staging.yml
  const stagingComposePath = path.join(projectRoot, 'docker-compose.staging.yml');
  console.log(`📄 Validating ${stagingComposePath}`);
  const stagingComposeConfig = parseDockerCompose(stagingComposePath);
  if (stagingComposeConfig) {
    const stagingResult = validateDockerCompose(
      stagingComposePath,
      stagingComposeConfig,
      envExampleVars,
      true
    );
    results.push(stagingResult);
    console.log(`   Found ${stagingComposeConfig.services.size} service overrides\n`);
  } else {
    results.push({
      file: stagingComposePath,
      valid: true, // Staging file is optional
      errors: [],
      warnings: ['docker-compose.staging.yml not found'],
    });
  }

  // Validate Dockerfile
  const dockerfilePath = path.join(projectRoot, 'Dockerfile');
  console.log(`📄 Validating ${dockerfilePath}`);
  const dockerfileResult = validateDockerfile(dockerfilePath);
  results.push(dockerfileResult);
  console.log('');

  // Validate env.ts schema
  const envSchemaPath = path.join(projectRoot, 'src/server/config/env.ts');
  console.log(`📄 Cross-validating ${envSchemaPath} against .env.example`);
  const schemaResult = validateEnvSchema(envExampleVars, envSchemaPath);
  results.push(schemaResult);
  console.log('');

  // Validate .env.staging configuration
  const stagingEnvPath = path.join(projectRoot, '.env.staging');
  console.log(`📄 Validating ${stagingEnvPath} for Phase 1 compliance`);
  const stagingEnvResult = validateStagingEnv(stagingEnvPath);
  results.push(stagingEnvResult);
  console.log('');

  // Validate CI workflow
  const ciWorkflowPath = path.join(projectRoot, '.github/workflows/ci.yml');
  console.log(`📄 Validating ${ciWorkflowPath} for orchestrator jobs`);
  const ciResult = validateCIWorkflow(ciWorkflowPath);
  results.push(ciResult);
  console.log('');

  return results;
}

// ============================================================================
// Output
// ============================================================================

function printResults(results: ValidationResult[]): boolean {
  let hasErrors = false;
  let totalErrors = 0;
  let totalWarnings = 0;

  console.log('\n' + '='.repeat(60));
  console.log('VALIDATION RESULTS');
  console.log('='.repeat(60) + '\n');

  for (const result of results) {
    const status = result.valid ? '✅' : '❌';
    console.log(`${status} ${result.file}`);

    for (const error of result.errors) {
      console.log(`   ❌ ERROR: ${error}`);
      totalErrors++;
      hasErrors = true;
    }

    for (const warning of result.warnings) {
      console.log(`   ⚠️  WARNING: ${warning}`);
      totalWarnings++;
    }

    if (result.errors.length === 0 && result.warnings.length === 0) {
      console.log('   All checks passed');
    }

    console.log('');
  }

  console.log('='.repeat(60));
  console.log(`SUMMARY: ${totalErrors} errors, ${totalWarnings} warnings`);
  console.log('='.repeat(60) + '\n');

  if (hasErrors) {
    console.log('❌ Validation FAILED - please fix errors before deploying\n');
  } else if (totalWarnings > 0) {
    console.log('⚠️  Validation PASSED with warnings - review before deploying\n');
  } else {
    console.log('✅ Validation PASSED - deployment configs are consistent\n');
  }

  return !hasErrors;
}

// ============================================================================
// Entry Point
// ============================================================================

export function validateDeploymentConfigProgrammatically(): DeploymentConfigValidationResult {
  const results = validateDeploymentConfigs();

  const errors: string[] = [];
  const warnings: string[] = [];

  for (const result of results) {
    if (result.errors.length > 0) {
      for (const error of result.errors) {
        errors.push(`[${result.file}] ${error}`);
      }
    }
    if (result.warnings.length > 0) {
      for (const warning of result.warnings) {
        warnings.push(`[${result.file}] ${warning}`);
      }
    }
  }

  const ok = results.every((r) => r.valid);

  return {
    ok,
    errors,
    warnings,
    results,
  };
}

function main(): void {
  const { results } = validateDeploymentConfigProgrammatically();
  const success = printResults(results);
  process.exit(success ? 0 : 1);
}

if (require.main === module) {
  main();
}
