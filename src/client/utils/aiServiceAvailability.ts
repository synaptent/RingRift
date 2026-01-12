/**
 * Utility for detecting whether AI service endpoints are available.
 *
 * In production (HTTPS or non-localhost), the AI service endpoints
 * (sandbox/ai/move, sandbox/evaluate, etc.) are not available unless
 * RINGRIFT_AI_SERVICE_URL is explicitly configured.
 *
 * This prevents 404 console errors when the client tries to call
 * AI service endpoints that don't exist on the production web server.
 */

import { readEnv } from '../../shared/utils/envFlags';

/**
 * Check if the sandbox AI service endpoints should be called.
 *
 * Returns false in production without RINGRIFT_AI_SERVICE_URL configured,
 * which prevents 404 errors from calling AI endpoints that don't exist
 * on the production web server.
 *
 * Returns true in:
 * - Local development (localhost or 127.0.0.1)
 * - When RINGRIFT_AI_SERVICE_URL is explicitly set
 */
export function isSandboxAIServiceAvailable(): boolean {
  // If RINGRIFT_AI_SERVICE_URL is set, the AI service is explicitly configured
  const envUrl = readEnv('RINGRIFT_AI_SERVICE_URL');
  if (envUrl && typeof envUrl === 'string') {
    return true;
  }

  // In production without configured URL, AI service is not available
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    const isProduction =
      protocol === 'https:' || (!hostname.includes('localhost') && hostname !== '127.0.0.1');
    if (isProduction) {
      return false;
    }
  }

  // Default: available in local development
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
