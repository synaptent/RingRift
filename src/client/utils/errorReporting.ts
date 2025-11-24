/**
 * Minimal, env-gated client-side error reporting utilities.
 *
 * This module intentionally keeps configuration surface small and
 * avoids any third-party SDKs. When disabled, all helpers are
 * cheap no-ops so they are safe to import in any client code.
 */
export interface ClientErrorContext {
  [key: string]: unknown;
}

interface NormalizedError {
  name: string;
  message: string;
  stack?: string;
}

interface ClientErrorPayload extends NormalizedError {
  context?: ClientErrorContext;
  url?: string;
  userAgent?: string;
  timestamp: string;
  type: string;
}

const env = (import.meta as any).env ?? {};

const ERROR_REPORTING_ENABLED: boolean = env.VITE_ERROR_REPORTING_ENABLED === 'true';
const ERROR_REPORTING_ENDPOINT: string =
  (env.VITE_ERROR_REPORTING_ENDPOINT as string | undefined) || '/api/client-errors';
const MAX_REPORTS_PER_SESSION: number = Number(env.VITE_ERROR_REPORTING_MAX_EVENTS ?? 50);

let reportsSent = 0;
let globalHandlersInstalled = false;

export function isErrorReportingEnabled(): boolean {
  return ERROR_REPORTING_ENABLED;
}

function shouldReport(): boolean {
  if (!ERROR_REPORTING_ENABLED) return false;
  if (reportsSent >= MAX_REPORTS_PER_SESSION) return false;
  reportsSent += 1;
  return true;
}

function normalizeError(error: unknown): NormalizedError {
  if (error instanceof Error) {
    return {
      name: error.name || 'Error',
      message: error.message || 'Unknown error',
      stack: error.stack,
    };
  }

  if (typeof error === 'string') {
    return {
      name: 'Error',
      message: error,
    };
  }

  try {
    const serialized = JSON.stringify(error);
    return {
      name: 'Error',
      message: serialized,
    };
  } catch {
    return {
      name: 'Error',
      message: String(error),
    };
  }
}

function buildPayload(error: unknown, context?: ClientErrorContext): ClientErrorPayload {
  const normalized = normalizeError(error);

  let url: string | undefined;
  let userAgent: string | undefined;

  if (typeof window !== 'undefined') {
    url = window.location?.href;
  }

  if (typeof navigator !== 'undefined') {
    userAgent = navigator.userAgent;
  }

  return {
    ...normalized,
    context,
    url,
    userAgent,
    timestamp: new Date().toISOString(),
    type: (context as any)?.type ?? 'client_error',
  };
}

export async function reportClientError(
  error: unknown,
  context?: ClientErrorContext
): Promise<void> {
  if (!shouldReport()) return;

  // Avoid throwing from the reporter; failures are best-effort only.
  try {
    if (typeof fetch !== 'function') {
      return;
    }

    const payload = buildPayload(error, context);

    // Fire-and-forget; callers do not await diagnostics.
    void fetch(ERROR_REPORTING_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      keepalive: true,
    });
  } catch (e) {
    // Swallow any errors. Optionally emit a console warning in dev builds.
    if ((env.MODE as string | undefined) === 'development') {
      // eslint-disable-next-line no-console
      console.warn('Failed to report client error', e);
    }
  }
}

/**
 * Attach window-level listeners for uncaught errors and unhandled promise
 * rejections. Safe to call multiple times; handlers are installed once.
 */
export function setupGlobalErrorHandlers(): void {
  if (!ERROR_REPORTING_ENABLED) return;
  if (typeof window === 'undefined' || typeof window.addEventListener !== 'function') {
    return;
  }
  if (globalHandlersInstalled) return;
  globalHandlersInstalled = true;

  window.addEventListener('error', (event) => {
    const anyEvent = event as any;
    // Ignore script load errors and similar noise; focus on actual Error objects.
    const error = anyEvent?.error || anyEvent?.message || 'Unknown window error';
    void reportClientError(error, { type: 'window_error' });
  });

  window.addEventListener('unhandledrejection', (event) => {
    const anyEvent = event as any;
    void reportClientError(anyEvent?.reason, { type: 'unhandledrejection' });
  });
}
