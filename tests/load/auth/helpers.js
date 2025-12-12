import http from 'k6/http';
import { check } from 'k6';

const DEFAULT_TOKEN_TTL_SECONDS = Number(__ENV.LOADTEST_AUTH_TOKEN_TTL_S || 15 * 60);
const TOKEN_REFRESH_SAFETY_WINDOW_SECONDS = Number(
  __ENV.LOADTEST_AUTH_REFRESH_WINDOW_S || 60
);

/**
 * Simple per-VU cache of the most recent successful login.
 * k6 executes each VU in its own JS runtime, so this state is per-VU.
 */
let cachedAuthState = null;

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
  const apiPrefix = (options && options.apiPrefix) || '/api';
  const tags = (options && options.tags) || { name: 'auth-login' };
  const metrics = (options && options.metrics) || {};
  const contractFailures = metrics.contractFailures;
  const capacityFailures = metrics.capacityFailures;

  const email = __ENV.LOADTEST_EMAIL || 'loadtest_user_1@loadtest.local';
  const password = __ENV.LOADTEST_PASSWORD || 'TestPassword123!';

  let res = http.post(
    `${baseUrl}${apiPrefix}/auth/login`,
    JSON.stringify({ email, password }),
    {
      headers: { 'Content-Type': 'application/json' },
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
        headers: { 'Content-Type': 'application/json' },
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

  // Update the per-VU auth cache so scenarios that call getValidToken(...)
  // can reuse this login and refresh it when nearing expiry.
  const obtainedAtMs = Date.now();
  const ttlSeconds =
    typeof expiresInSeconds === 'number' && expiresInSeconds > 0
      ? expiresInSeconds
      : DEFAULT_TOKEN_TTL_SECONDS;

  cachedAuthState = {
    baseUrl,
    token: accessToken,
    userId,
    obtainedAtMs,
    expiresAtMs: obtainedAtMs + ttlSeconds * 1000,
    ttlSeconds,
  };

  return { token: accessToken, userId, expiresInSeconds };
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

  const shouldLogin =
    forceRefresh ||
    !cachedAuthState ||
    cachedAuthState.baseUrl !== baseUrl ||
    // Proactively refresh shortly before the token is expected to expire.
    now >= cachedAuthState.expiresAtMs - TOKEN_REFRESH_SAFETY_WINDOW_SECONDS * 1000;

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
