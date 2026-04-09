/**
 * Utility for detecting whether AI service endpoints are available.
 *
 * The sandbox AI endpoints (/api/games/sandbox/ai/move, etc.) are
 * served by the same Express backend, which proxies to the Python AI
 * service when AI_SERVICE_URL is configured server-side.
 *
 * Previously this returned false in production to prevent 404 errors,
 * but the production server now has AI_SERVICE_URL configured and the
 * sandbox endpoints enabled. Returning true allows the sandbox and
 * lobby to use neural network AI in production.
 *
 * Set RINGRIFT_AI_SERVICE_URL in the client environment to override
 * the backend URL (e.g., for direct client→AI-service testing).
 */

import { readEnv } from '../../shared/utils/envFlags';

/**
 * Check if the sandbox AI service endpoints should be called.
 *
 * Returns true in all environments — the backend handles availability
 * and falls back to heuristic AI if the Python service is unreachable.
 */
export function isSandboxAIServiceAvailable(): boolean {
  return true;
}

/**
 * Cached result of the availability check.
 * Computed once on first call to avoid repeated DOM/window checks.
 */
let cachedAvailability: boolean | null = null;

/**
 * Get cached sandbox AI service availability.
 * Uses cached result after first check for performance.
 */
export function getSandboxAIServiceAvailable(): boolean {
  if (cachedAvailability === null) {
    cachedAvailability = isSandboxAIServiceAvailable();
  }
  return cachedAvailability;
}

/**
 * Reset the cached availability (for testing).
 */
export function resetSandboxAIServiceAvailabilityCache(): void {
  cachedAvailability = null;
}
