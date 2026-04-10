/// <reference types="vite/client" />

/**
 * Extend Vite's built-in ImportMetaEnv with custom VITE_* environment variables.
 * Vite already defines MODE, DEV, PROD, SSR in its client types.
 * We only add our application-specific variables here.
 */
interface ImportMetaEnv {
  readonly VITE_ERROR_REPORTING_ENABLED?: string | undefined;
  readonly VITE_ERROR_REPORTING_ENDPOINT?: string | undefined;
  readonly VITE_ERROR_REPORTING_MAX_EVENTS?: string | undefined;
  readonly VITE_API_URL?: string | undefined;
  readonly VITE_WS_URL?: string | undefined;
  readonly VITE_SENTRY_DSN?: string | undefined;
  readonly VITE_DIFFICULTY_CALIBRATION_TELEMETRY_ENABLED?: string | undefined;
  readonly VITE_RULES_UX_TELEMETRY_ENABLED?: string | undefined;
  readonly VITE_RULES_UX_HELP_OPEN_SAMPLE_RATE?: string | undefined;
  readonly VITE_CLIENT_BUILD?: string | undefined;
  readonly VITE_GIT_SHA?: string | undefined;
  readonly VITE_APP_VERSION?: string | undefined;
}
