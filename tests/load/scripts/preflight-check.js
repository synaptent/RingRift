#!/usr/bin/env node
/**
 * RingRift Load Test Pre-Flight Health Check
 *
 * Validates all required dependencies are operational before starting load tests.
 * This prevents wasted test runs due to infrastructure issues.
 *
 * Usage:
 *   node tests/load/scripts/preflight-check.js [options]
 *
 * Options:
 *   --skip-ai          Skip AI service health check (for tests not requiring AI)
 *   --skip-user-pool   Skip user pool validation
 *   --min-users N      Minimum users required in pool (default: 10)
 *   --expected-vus N   Expected peak VUs (warn if user pool is smaller)
 *   --expected-duration-s N  Expected test duration in seconds (auth TTL warnings)
 *   --verbose          Show detailed output for each check
 *   --json             Output results as JSON
 *
 * Environment Variables:
 *   BASE_URL           Backend API base URL (default: http://localhost:3000)
 *   AI_SERVICE_URL     AI service URL (default: http://localhost:8001)
 *   LOADTEST_EMAIL     Test user email (default: loadtest_user_1@loadtest.local)
 *   LOADTEST_PASSWORD  Test user password (default: LoadTestK6Pass123)
 *   LOADTEST_USER_POOL_SIZE  Number of users in pool (default: 0)
 *   LOADTEST_USER_PASSWORD   Password used by the seeding script (optional)
 *   LOADTEST_EXPECTED_VUS    Expected peak VUs (optional)
 *   LOADTEST_EXPECTED_DURATION_S  Expected test duration in seconds (optional)
 *   LOADTEST_AUTH_TOKEN_TTL_S      Token TTL override when expiresIn is missing (optional)
 *   LOADTEST_AUTH_REFRESH_WINDOW_S Token refresh safety window (optional)
 */

const http = require('http');
const https = require('https');

// ============================================================================
// Configuration
// ============================================================================

const userPoolSize = parseInt(process.env.LOADTEST_USER_POOL_SIZE || '0', 10);
const userPoolPrefix = process.env.LOADTEST_USER_POOL_PREFIX || 'loadtest_user_';
const userPoolDomain = process.env.LOADTEST_USER_POOL_DOMAIN || 'loadtest.local';
const userPoolPassword = process.env.LOADTEST_USER_POOL_PASSWORD || 'LoadTestK6Pass123';
const hasExplicitTestEmail = Object.prototype.hasOwnProperty.call(process.env, 'LOADTEST_EMAIL');
const hasExplicitTestPassword = Object.prototype.hasOwnProperty.call(process.env, 'LOADTEST_PASSWORD');

const defaultTestEmail = 'loadtest_user_1@loadtest.local';
const poolTestEmail = `${userPoolPrefix}1@${userPoolDomain}`;

const config = {
  baseUrl: process.env.BASE_URL || 'http://localhost:3000',
  aiServiceUrl: process.env.AI_SERVICE_URL || 'http://localhost:8001',
  testEmail: hasExplicitTestEmail
    ? process.env.LOADTEST_EMAIL
    : userPoolSize > 0
      ? poolTestEmail
      : defaultTestEmail,
  testPassword: hasExplicitTestPassword
    ? process.env.LOADTEST_PASSWORD
    : userPoolSize > 0
      ? userPoolPassword
      : 'LoadTestK6Pass123',
  userPoolSize,
  userPoolPrefix,
  userPoolDomain,
  userPoolPassword,
  seedUserPassword: process.env.LOADTEST_USER_PASSWORD,
  expectedVus: parseInt(process.env.LOADTEST_EXPECTED_VUS || '0', 10),
  expectedDurationSeconds: parseInt(
    process.env.LOADTEST_EXPECTED_DURATION_S || '0',
    10
  ),
  authTokenTtlOverrideSeconds: parseInt(
    process.env.LOADTEST_AUTH_TOKEN_TTL_S || '0',
    10
  ),
  authRefreshWindowSeconds: parseInt(
    process.env.LOADTEST_AUTH_REFRESH_WINDOW_S || '60',
    10
  ),
};

// Parse command line arguments
const args = process.argv.slice(2);
function parseIntFlag(flag) {
  const idx = args.indexOf(flag);
  if (idx === -1) return null;
  const raw = args[idx + 1];
  if (!raw) return null;
  const value = parseInt(raw, 10);
  return Number.isFinite(value) ? value : null;
}

const expectedVusArg = parseIntFlag('--expected-vus');
const expectedDurationArg = parseIntFlag('--expected-duration-s');

const options = {
  skipAi: args.includes('--skip-ai'),
  skipUserPool: args.includes('--skip-user-pool'),
  minUsers: parseInt(args.find((a, i) => args[i - 1] === '--min-users') || '10', 10),
  verbose: args.includes('--verbose'),
  json: args.includes('--json'),
  expectedVus:
    expectedVusArg !== null
      ? expectedVusArg
      : config.expectedVus > 0
        ? config.expectedVus
        : null,
  expectedDurationSeconds:
    expectedDurationArg !== null
      ? expectedDurationArg
      : config.expectedDurationSeconds > 0
        ? config.expectedDurationSeconds
        : null,
};

// ============================================================================
// Utility Functions
// ============================================================================

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m',
};

function log(message) {
  if (!options.json) {
    console.log(message);
  }
}

function logVerbose(message) {
  if (options.verbose && !options.json) {
    console.log(`${colors.dim}  ${message}${colors.reset}`);
  }
}

function pass(name, details) {
  log(`${colors.green}✅ ${name}: PASS${colors.reset}${details ? ` ${colors.dim}(${details})${colors.reset}` : ''}`);
}

function fail(name, error) {
  log(`${colors.red}❌ ${name}: FAIL${colors.reset} - ${error}`);
}

function skip(name, reason) {
  log(`${colors.yellow}⏭️  ${name}: SKIPPED${colors.reset} - ${reason}`);
}

function warn(name, message) {
  log(`${colors.yellow}⚠️  ${name}: WARNING${colors.reset} - ${message}`);
}

/**
 * Make an HTTP/HTTPS request
 * @param {string} url - Full URL to request
 * @param {Object} options - Request options
 * @returns {Promise<{status: number, body: string, headers: Object}>}
 */
function httpRequest(url, requestOptions = {}) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);
    const client = parsedUrl.protocol === 'https:' ? https : http;

    const reqOptions = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (parsedUrl.protocol === 'https:' ? 443 : 80),
      path: parsedUrl.pathname + parsedUrl.search,
      method: requestOptions.method || 'GET',
      headers: requestOptions.headers || {},
      timeout: requestOptions.timeout || 10000,
    };

    const req = client.request(reqOptions, (res) => {
      let body = '';
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        resolve({
          status: res.statusCode,
          body,
          headers: res.headers,
        });
      });
    });

    req.on('error', (error) => {
      reject(new Error(`Request failed: ${error.message}`));
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timed out'));
    });

    if (requestOptions.body) {
      req.write(requestOptions.body);
    }

    req.end();
  });
}

// ============================================================================
// Health Checks
// ============================================================================

const checks = {
  /**
   * Validate backend health endpoint
   */
  async backendHealth() {
    const url = `${config.baseUrl}/health`;
    logVerbose(`Checking ${url}`);

    const response = await httpRequest(url);

    if (response.status !== 200) {
      throw new Error(`Health endpoint returned ${response.status}`);
    }

    let healthData;
    try {
      healthData = JSON.parse(response.body);
    } catch {
      // Non-JSON response is acceptable if status is 200
      healthData = { status: 'ok' };
    }

    const details = [];
    if (healthData.status) details.push(`status=${healthData.status}`);
    if (healthData.version) details.push(`version=${healthData.version}`);
    if (healthData.uptime) details.push(`uptime=${healthData.uptime}s`);

    return {
      success: true,
      details: details.join(', ') || 'healthy',
      data: healthData,
    };
  },

  /**
   * Validate load-test configuration consistency (user pool + auth settings)
   */
  async loadTestConfig() {
    const warnings = [];
    const details = [];

    const expectedVus = options.expectedVus || 0;
    const expectedDurationSeconds = options.expectedDurationSeconds || 0;

    details.push(`user_pool_size=${config.userPoolSize}`);
    if (expectedVus > 0) {
      details.push(`expected_vus=${expectedVus}`);
    }
    if (expectedDurationSeconds > 0) {
      details.push(`expected_duration_s=${expectedDurationSeconds}`);
    }

    if (expectedVus > 1) {
      if (config.userPoolSize === 0) {
        warnings.push(
          `LOADTEST_USER_POOL_SIZE is 0 with expected_vus=${expectedVus}; enable a pool to avoid per-user rate limits`
        );
      } else if (config.userPoolSize < expectedVus) {
        warnings.push(
          `LOADTEST_USER_POOL_SIZE (${config.userPoolSize}) < expected_vus (${expectedVus}); increase pool size`
        );
      }
    }

    const shouldUsePool = config.userPoolSize > 0 || expectedVus > 1;
    if (
      shouldUsePool &&
      config.seedUserPassword &&
      config.userPoolPassword &&
      config.seedUserPassword !== config.userPoolPassword
    ) {
      warnings.push(
        'LOADTEST_USER_PASSWORD differs from LOADTEST_USER_POOL_PASSWORD; align seeding and k6 pool credentials'
      );
    }

    const data = {
      expectedVus,
      expectedDurationSeconds,
      userPoolSize: config.userPoolSize,
      hasSeedUserPassword: Boolean(config.seedUserPassword),
    };

    if (warnings.length > 0) {
      return {
        success: true,
        warning: true,
        details: warnings.join(' | '),
        data,
      };
    }

    return {
      success: true,
      details: details.join(', ') || 'config ok',
      data,
    };
  },

  /**
   * Validate AI service health endpoint
   */
  async aiServiceHealth() {
    if (options.skipAi) {
      return { skipped: true, reason: '--skip-ai flag provided' };
    }

    const url = `${config.aiServiceUrl}/health`;
    logVerbose(`Checking ${url}`);

    try {
      const response = await httpRequest(url, { timeout: 5000 });

      if (response.status !== 200) {
        throw new Error(`AI service returned ${response.status}`);
      }

      let healthData;
      try {
        healthData = JSON.parse(response.body);
      } catch {
        healthData = { status: 'ok' };
      }

      const details = [];
      if (healthData.status) details.push(`status=${healthData.status}`);
      if (healthData.version) details.push(`version=${healthData.version}`);
      if (healthData.model_loaded !== undefined) details.push(`model_loaded=${healthData.model_loaded}`);

      return {
        success: true,
        details: details.join(', ') || 'healthy',
        data: healthData,
      };
    } catch (error) {
      // AI service is optional by default - return warning instead of failure
      return {
        success: true,
        warning: true,
        details: `AI service unavailable: ${error.message}`,
      };
    }
  },

  /**
   * Validate auth system - login with test user and verify token with expiresIn
   */
  async authSystem() {
    const loginUrl = `${config.baseUrl}/api/auth/login`;
    logVerbose(`Attempting login at ${loginUrl}`);

    const payload = JSON.stringify({
      email: config.testEmail,
      password: config.testPassword,
    });

    let response;
    try {
      response = await httpRequest(loginUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: payload,
      });
    } catch (error) {
      throw new Error(`Login request failed: ${error.message}`);
    }

    // Handle 404 - try with trailing slash (some deployments require it)
    if (response.status === 404) {
      logVerbose('Retrying with trailing slash...');
      response = await httpRequest(`${loginUrl}/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: payload,
      });
    }

    if (response.status !== 200) {
      throw new Error(`Login failed with status ${response.status}: ${response.body}`);
    }

    let loginData;
    try {
      loginData = JSON.parse(response.body);
    } catch {
      throw new Error('Invalid JSON response from login endpoint');
    }

    // Extract token from response envelope
    const data = loginData.data || loginData;
    const accessToken = data.accessToken;
    const expiresIn = data.expiresIn;
    const user = data.user;
    const warnings = [];

    if (!accessToken || typeof accessToken !== 'string') {
      throw new Error('No accessToken returned from login');
    }

    const hasExpiresIn = typeof expiresIn === 'number' && expiresIn > 0;
    let ttlSeconds = hasExpiresIn ? expiresIn : null;

    if (!hasExpiresIn) {
      if (config.authTokenTtlOverrideSeconds > 0) {
        ttlSeconds = config.authTokenTtlOverrideSeconds;
        warnings.push(
          `expiresIn missing; using LOADTEST_AUTH_TOKEN_TTL_S=${ttlSeconds}s`
        );
      } else {
        throw new Error('expiresIn field missing from login response (required per PV-02)');
      }
    }

    if (hasExpiresIn && config.authTokenTtlOverrideSeconds > 0) {
      const delta = Math.abs(expiresIn - config.authTokenTtlOverrideSeconds);
      const deltaRatio = delta / expiresIn;
      if (deltaRatio > 0.1) {
        warnings.push(
          `LOADTEST_AUTH_TOKEN_TTL_S (${config.authTokenTtlOverrideSeconds}s) differs from expiresIn (${expiresIn}s)`
        );
      }
    }

    if (ttlSeconds !== null && config.authRefreshWindowSeconds >= ttlSeconds) {
      warnings.push(
        `LOADTEST_AUTH_REFRESH_WINDOW_S (${config.authRefreshWindowSeconds}s) >= token TTL (${ttlSeconds}s)`
      );
    }

    if (
      ttlSeconds !== null &&
      options.expectedDurationSeconds &&
      ttlSeconds < options.expectedDurationSeconds
    ) {
      warnings.push(
        `Token TTL (${ttlSeconds}s) < expected_duration_s (${options.expectedDurationSeconds}s); refreshes required`
      );
    }

    const details = [];
    details.push(`token_length=${accessToken.length}`);
    if (hasExpiresIn) {
      details.push(`expiresIn=${expiresIn}s`);
    } else if (ttlSeconds !== null) {
      details.push(`expiresIn=missing (override=${ttlSeconds}s)`);
    }
    details.push(`refresh_window=${config.authRefreshWindowSeconds}s`);
    if (user && user.id) details.push(`userId=${user.id.substring(0, 8)}...`);

    const detailText = warnings.length > 0
      ? `${details.join(', ')}; ${warnings.join('; ')}`
      : details.join(', ');

    return {
      success: true,
      warning: warnings.length > 0,
      details: detailText,
      data: {
        hasToken: true,
        expiresIn,
        ttlSeconds,
        refreshWindowSeconds: config.authRefreshWindowSeconds,
        warnings,
        hasUser: !!user,
      },
    };
  },

  /**
   * Validate user pool - check expected users exist and can authenticate
   */
  async userPool() {
    if (options.skipUserPool) {
      return { skipped: true, reason: '--skip-user-pool flag provided' };
    }

    if (config.userPoolSize === 0) {
      return { skipped: true, reason: 'LOADTEST_USER_POOL_SIZE not configured' };
    }

    logVerbose(`Validating user pool with ${config.userPoolSize} expected users`);

    const usersToCheck = Math.min(options.minUsers, config.userPoolSize);
    const results = { successful: 0, failed: 0, errors: [] };

    // Check a sample of users (not all, to keep preflight fast)
    const checkIndices = [];
    for (let i = 0; i < usersToCheck; i++) {
      // Spread checks across pool: first, last, and evenly distributed
      const index = Math.floor((i / (usersToCheck - 1 || 1)) * (config.userPoolSize - 1)) + 1;
      if (!checkIndices.includes(index)) {
        checkIndices.push(index);
      }
    }

    for (const userIndex of checkIndices) {
      const email = `${config.userPoolPrefix}${userIndex}@${config.userPoolDomain}`;
      logVerbose(`Checking user ${email}`);

      try {
        const response = await httpRequest(`${config.baseUrl}/api/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password: config.userPoolPassword }),
          timeout: 5000,
        });

        if (response.status === 200) {
          results.successful++;
        } else {
          results.failed++;
          results.errors.push(`${email}: status ${response.status}`);
        }
      } catch (error) {
        results.failed++;
        results.errors.push(`${email}: ${error.message}`);
      }
    }

    if (results.successful < options.minUsers && results.failed > 0) {
      throw new Error(
        `Only ${results.successful}/${checkIndices.length} pool users authenticated. ` +
        `First error: ${results.errors[0]}`
      );
    }

    return {
      success: true,
      details: `${results.successful}/${checkIndices.length} users validated (pool size: ${config.userPoolSize})`,
      data: results,
    };
  },
};

// ============================================================================
// Main Execution
// ============================================================================

async function runPreflight() {
  const startTime = Date.now();
  log('');
  log(`${colors.cyan}🔍 Running load test preflight checks...${colors.reset}`);
  log(`${colors.dim}   Target: ${config.baseUrl}${colors.reset}`);
  log('');

  const results = {
    timestamp: new Date().toISOString(),
    baseUrl: config.baseUrl,
    aiServiceUrl: config.aiServiceUrl,
    checks: {},
  };

  const checkOrder = [
    'loadTestConfig',
    'backendHealth',
    'aiServiceHealth',
    'authSystem',
    'userPool',
  ];
  const failures = [];
  const warnings = [];

  for (const name of checkOrder) {
    const check = checks[name];
    try {
      const result = await check();

      if (result.skipped) {
        skip(name, result.reason);
        results.checks[name] = { status: 'SKIPPED', reason: result.reason };
      } else if (result.warning) {
        warn(name, result.details);
        warnings.push({ name, message: result.details });
        results.checks[name] = { status: 'WARNING', details: result.details };
      } else {
        pass(name, result.details);
        results.checks[name] = { status: 'PASS', details: result.details, data: result.data };
      }
    } catch (error) {
      fail(name, error.message);
      failures.push({ name, error: error.message });
      results.checks[name] = { status: 'FAIL', error: error.message };

      // Exit early on critical failure (backend or auth)
      if (name === 'backendHealth' || name === 'authSystem') {
        log('');
        log(`${colors.red}💥 Critical check failed - aborting preflight${colors.reset}`);
        break;
      }
    }
  }

  const duration = Date.now() - startTime;
  results.duration_ms = duration;
  results.passed = failures.length === 0;
  results.failure_count = failures.length;
  results.warning_count = warnings.length;

  log('');

  if (options.json) {
    console.log(JSON.stringify(results, null, 2));
  }

  if (failures.length > 0) {
    log(`${colors.red}❌ Preflight failed: ${failures.length} check(s) failed${colors.reset}`);
    if (!options.json) {
      for (const f of failures) {
        log(`   - ${f.name}: ${f.error}`);
      }
    }
    log(`${colors.dim}   Duration: ${duration}ms${colors.reset}`);
    log('');
    process.exit(1);
  }

  if (warnings.length > 0) {
    log(`${colors.yellow}⚠️  Preflight passed with ${warnings.length} warning(s)${colors.reset}`);
  } else {
    log(`${colors.green}✅ All preflight checks passed${colors.reset}`);
  }
  log(`${colors.dim}   Duration: ${duration}ms${colors.reset}`);
  log('');

  process.exit(0);
}

// Run the preflight checks
runPreflight().catch((error) => {
  console.error(`${colors.red}Unexpected error: ${error.message}${colors.reset}`);
  process.exit(1);
});
