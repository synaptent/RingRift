import http from 'k6/http';
import { check } from 'k6';
import encoding from 'k6/encoding';

const DEFAULT_TOKEN_TTL_SECONDS = Number(__ENV.LOADTEST_AUTH_TOKEN_TTL_S || 15 * 60);
const TOKEN_REFRESH_SAFETY_WINDOW_SECONDS = Number(
  __ENV.LOADTEST_AUTH_REFRESH_WINDOW_S || 60
);
// Maximum jitter (in seconds) added to token refresh timing to prevent "refresh storms"
// when many VUs attempt to refresh simultaneously
const TOKEN_REFRESH_JITTER_MAX_SECONDS = Number(
  __ENV.LOADTEST_AUTH_REFRESH_JITTER_S || 30
);

function decodeBase64Url(input) {
  if (!input) return null;
  try {
    return encoding.b64decode(input, 'rawurl', 's');
  } catch (error) {
    try {
      const normalized = input.replace(/-/g, '+').replace(/_/g, '/');
      const padding = normalized.length % 4;
      const padded = padding ? `${normalized}${'='.repeat(4 - padding)}` : normalized;
      return encoding.b64decode(padded, 'rawstd', 's');
    } catch (fallbackError) {
      return null;
    }
  }
}

function deriveJwtTtlSeconds(token) {
  if (!token || typeof token !== 'string') return null;
  const parts = token.split('.');
  if (parts.length < 2) return null;
  const payload = decodeBase64Url(parts[1]);
  if (!payload) return null;
  try {
    const data = JSON.parse(payload);
    if (!data || typeof data.exp !== 'number') return null;
    const nowSeconds = Math.floor(Date.now() / 1000);
    const ttlSeconds = data.exp - nowSeconds;
    return ttlSeconds > 0 ? ttlSeconds : null;
  } catch (error) {
    return null;
  }
}

/**
 * Rate limit bypass token.
 * When set, this token is sent in the X-RateLimit-Bypass-Token header on all
 * HTTP requests to bypass rate limiting during load tests.
 *
 * Environment variable:
 *   RATE_LIMIT_BYPASS_TOKEN - Must match the server's RATE_LIMIT_BYPASS_TOKEN
 */
const RATE_LIMIT_BYPASS_TOKEN = __ENV.RATE_LIMIT_BYPASS_TOKEN || '';

/**
 * Get headers that include the rate limit bypass token if configured.
 * @param {Object} [additionalHeaders] - Additional headers to merge
 * @returns {Object} Headers object with bypass token if configured
 */
export function getBypassHeaders(additionalHeaders = {}) {
  const headers = { ...additionalHeaders };
  if (RATE_LIMIT_BYPASS_TOKEN) {
    headers['X-RateLimit-Bypass-Token'] = RATE_LIMIT_BYPASS_TOKEN;
  }
  return headers;
}

/**
 * Multi-user pool configuration.
 *
 * When LOADTEST_USER_POOL_SIZE is set (e.g., 400), users will be selected
 * from loadtest_user_1@loadtest.local through loadtest_user_N@loadtest.local
 * based on the VU number. This distributes load across multiple users to
 * avoid per-user rate limits.
 *
 * Environment variables:
 *   LOADTEST_USER_POOL_SIZE - Number of users in the pool (default: 0 = single user mode)
 *   LOADTEST_USER_POOL_PASSWORD - Shared password for all pool users (default: LoadTestK6Pass123)
 *   LOADTEST_USER_POOL_PREFIX - Email prefix pattern (default: loadtest_user_)
 *   LOADTEST_USER_POOL_DOMAIN - Email domain (default: loadtest.local)
 */
const USER_POOL_SIZE = Number(__ENV.LOADTEST_USER_POOL_SIZE || 0);
const USER_POOL_PASSWORD = __ENV.LOADTEST_USER_POOL_PASSWORD || 'LoadTestK6Pass123';
const USER_POOL_PREFIX = __ENV.LOADTEST_USER_POOL_PREFIX || 'loadtest_user_';
const USER_POOL_DOMAIN = __ENV.LOADTEST_USER_POOL_DOMAIN || 'loadtest.local';

/**
 * Get credentials for the current VU.
 * When USER_POOL_SIZE > 0, selects user based on VU number.
 * Otherwise falls back to single-user mode.
 *
 * @param {number} [vuOverride] - Optional VU number override (useful in setup phase)
 * @returns {{ email: string, password: string, userIndex: number }}
 */
export function getVUCredentials(vuOverride) {
  if (USER_POOL_SIZE > 0) {
    // __VU is 1-indexed, user pool is 1-indexed
    // During setup phase, __VU is 0, so default to user 1
    // Use modulo to wrap around if VUs exceed pool size
    const vu = typeof vuOverride === 'number' ? vuOverride : (__VU || 1);
    const userIndex = ((vu - 1) % USER_POOL_SIZE) + 1;
    return {
      email: `${USER_POOL_PREFIX}${userIndex}@${USER_POOL_DOMAIN}`,
      password: USER_POOL_PASSWORD,
      userIndex,
    };
  }

  // Single-user fallback mode - use same password as pool mode for consistency
  return {
    email: __ENV.LOADTEST_EMAIL || 'loadtest_user_1@loadtest.local',
    password: __ENV.LOADTEST_PASSWORD || 'LoadTestK6Pass123',
    userIndex: 1,
  };
}

/**
 * Simple per-VU cache of the most recent successful login.
 * k6 executes each VU in its own JS runtime, so this state is per-VU.
 */
let cachedAuthState = null;
const cachedAuthStatesByUser = new Map();

function buildAuthState(baseUrl, token, userId, expiresInSeconds) {
  const obtainedAtMs = Date.now();
  const derivedTtlSeconds =
    typeof expiresInSeconds === 'number' && expiresInSeconds > 0
      ? null
      : deriveJwtTtlSeconds(token);

  const ttlSeconds =
    typeof expiresInSeconds === 'number' && expiresInSeconds > 0
      ? expiresInSeconds
      : typeof derivedTtlSeconds === 'number' && derivedTtlSeconds > 0
        ? derivedTtlSeconds
        : DEFAULT_TOKEN_TTL_SECONDS;

  return {
    baseUrl,
    token,
    userId,
    obtainedAtMs,
    expiresAtMs: obtainedAtMs + ttlSeconds * 1000,
    ttlSeconds,
  };
}

function loginWithCredentials(baseUrl, credentials, options) {
  const apiPrefix = (options && options.apiPrefix) || '/api';
  const tags = (options && options.tags) || { name: 'auth-login' };
  const metrics = (options && options.metrics) || {};
  const contractFailures = metrics.contractFailures;
  const capacityFailures = metrics.capacityFailures;

  const email = credentials.email;
  const password = credentials.password;

  let res = http.post(
    `${baseUrl}${apiPrefix}/auth/login`,
    JSON.stringify({ email, password }),
    {
      headers: getBypassHeaders({ 'Content-Type': 'application/json' }),
      tags,
    }
  );

  // Some production/staging deployments enforce a trailing slash on auth routes.
  // Fall back to /auth/login/ when the non-slashed path returns 404.
  if (res.status === 404) {
    res = http.post(
      `${baseUrl}${apiPrefix}/auth/login/`,
      JSON.stringify({ email, password }),
      {
        headers: getBypassHeaders({ 'Content-Type': 'application/json' }),
        tags,
      }
    );
  }

  let parsed = null;
  let accessToken = null;
  let userId = null;
  let expiresInSeconds = null;

  try {
    parsed = JSON.parse(res.body);
    const data = parsed && parsed.data ? parsed.data : null;

    accessToken =
      data && typeof data.accessToken === 'string' ? data.accessToken : null;
    userId = data && data.user && data.user.id ? data.user.id : null;

    // Prefer an explicit expiresIn field when provided by the API.
    if (data && typeof data.expiresIn === 'number') {
      expiresInSeconds = data.expiresIn;
    }
  } catch (err) {
    parsed = null;
    accessToken = null;
    userId = null;
    expiresInSeconds = null;
  }

  const ok = check(res, {
    'login successful': (r) => r.status === 200,
    'access token present': () => typeof accessToken === 'string',
  });

  if (!ok) {
    // Classify login failures so we can distinguish contract vs capacity issues.
    if (!res || res.status === 0) {
      if (capacityFailures) capacityFailures.add(1);
    } else if (res.status >= 400 && res.status < 500 && res.status !== 429) {
      if (contractFailures) contractFailures.add(1);
    } else if (res.status === 429 || res.status >= 500) {
      if (capacityFailures) capacityFailures.add(1);
    } else if (res.status === 200 && typeof accessToken !== 'string') {
      // 200 with a malformed body is a contract failure.
      if (contractFailures) contractFailures.add(1);
    }

    throw new Error(`loginAndGetToken failed: status=${res.status} body=${res.body}`);
  }

  return { token: accessToken, userId, expiresInSeconds };
}

/**
 * Shared login helper for all RingRift k6 scenarios.
 *
 * This helper:
 * - Calls POST /api/auth/login with the canonical payload shape:
 *     { email, password }
 * - Allows overriding credentials via LOADTEST_EMAIL / LOADTEST_PASSWORD
 * - Returns { token, userId } on success
 *
 * It also optionally records classification metrics when provided via
 * options.metrics:
 *   - contractFailures (Counter: contract_failures_total)
 *   - capacityFailures (Counter: capacity_failures_total)
 *
 * Any scenario using this helper will fail fast if login cannot be
 * established, since meaningful load testing depends on authenticated
 * requests.
 *
 * @param {string} baseUrl - Base HTTP origin, e.g. http://localhost:3001
 * @param {Object} [options]
 * @param {string} [options.apiPrefix='/api'] - API prefix to use
 * @param {Object} [options.tags] - Optional k6 tags to attach to the login request
 * @param {{ contractFailures?: any, capacityFailures?: any }} [options.metrics] - Optional
 *   classification counters to record failures against.
 * @returns {{ token: string, userId: string | null }}
 */
export function loginAndGetToken(baseUrl, options) {
  // Use multi-user pool when configured, otherwise fall back to single user.
  const credentials = getVUCredentials();
  const result = loginWithCredentials(baseUrl, credentials, options);

  cachedAuthState = buildAuthState(
    baseUrl,
    result.token,
    result.userId,
    result.expiresInSeconds
  );

  return result;
}

export function loginAndGetTokenForUser(baseUrl, options) {
  const userIndex = options && options.userIndex;
  if (!userIndex) {
    return loginAndGetToken(baseUrl, options);
  }

  const credentials = getVUCredentials(userIndex);
  const result = loginWithCredentials(baseUrl, credentials, options);
  const cacheKey = `${baseUrl}:${userIndex}`;
  cachedAuthStatesByUser.set(
    cacheKey,
    buildAuthState(baseUrl, result.token, result.userId, result.expiresInSeconds)
  );

  return result;
}

/**
 * Obtain a currently-valid access token for load tests, refreshing it when it is
 * close to expiry or when a caller explicitly forces a refresh.
 *
 * This helper wraps loginAndGetToken and maintains a simple per-VU cache:
 * - The first call performs a login.
 * - Subsequent calls reuse the cached token until it is within
 *   TOKEN_REFRESH_SAFETY_WINDOW_SECONDS of its expiry, at which point a new
 *   login is performed.
 *
 * @param {string} baseUrl
 * @param {Object} [options]
 * @param {string} [options.apiPrefix='/api']
 * @param {Object} [options.tags]
 * @param {{ contractFailures?: any, capacityFailures?: any }} [options.metrics]
 * @param {boolean} [options.forceRefresh] - When true, always perform a fresh login.
 * @returns {{ token: string, userId: string | null }}
 */
export function getValidToken(baseUrl, options) {
  const forceRefresh = options && options.forceRefresh;
  const apiPrefix = (options && options.apiPrefix) || '/api';
  const tags = (options && options.tags) || { name: 'auth-login' };
  const metrics = (options && options.metrics) || {};

  const now = Date.now();

  // Add jitter to token refresh timing to prevent "refresh storms" when many VUs
  // reach the refresh window simultaneously. Each VU gets a random offset so
  // refreshes are distributed across the jitter window.
  const jitterMs = Math.random() * TOKEN_REFRESH_JITTER_MAX_SECONDS * 1000;

  const shouldLogin =
    forceRefresh ||
    !cachedAuthState ||
    cachedAuthState.baseUrl !== baseUrl ||
    // Proactively refresh shortly before the token is expected to expire,
    // with added jitter to spread out refresh requests.
    now >= cachedAuthState.expiresAtMs - TOKEN_REFRESH_SAFETY_WINDOW_SECONDS * 1000 - jitterMs;

  if (shouldLogin) {
    const result = loginAndGetToken(baseUrl, {
      apiPrefix,
      tags,
      metrics,
    });

    // loginAndGetToken already updated cachedAuthState; just mirror the
    // returned token and userId here for callers.
    return { token: result.token, userId: result.userId };
  }

  return { token: cachedAuthState.token, userId: cachedAuthState.userId };
}

export function getValidTokenForUser(baseUrl, options) {
  const userIndex = options && options.userIndex;
  if (!userIndex) {
    return getValidToken(baseUrl, options);
  }

  const forceRefresh = options && options.forceRefresh;
  const apiPrefix = (options && options.apiPrefix) || '/api';
  const tags = (options && options.tags) || { name: 'auth-login' };
  const metrics = (options && options.metrics) || {};

  const cacheKey = `${baseUrl}:${userIndex}`;
  const cached = cachedAuthStatesByUser.get(cacheKey);
  const now = Date.now();
  const jitterMs = Math.random() * TOKEN_REFRESH_JITTER_MAX_SECONDS * 1000;

  const shouldLogin =
    forceRefresh ||
    !cached ||
    cached.baseUrl !== baseUrl ||
    now >= cached.expiresAtMs - TOKEN_REFRESH_SAFETY_WINDOW_SECONDS * 1000 - jitterMs;

  if (shouldLogin) {
    const result = loginAndGetTokenForUser(baseUrl, {
      apiPrefix,
      tags,
      metrics,
      userIndex,
    });

    return { token: result.token, userId: result.userId };
  }

  return { token: cached.token, userId: cached.userId };
}

export function getUserPoolSize() {
  return USER_POOL_SIZE;
}
