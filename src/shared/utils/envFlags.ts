// Shared helpers for reading environment flags in both Node/Jest and the
// browser bundle. For browser builds, this relies on the bundler exposing
// process.env-style variables (for example via Vite define/plugin wiring).
// Keeping this logic centralised ensures sandbox AI diagnostics behave
// consistently across environments.

export function readEnv(name: string): string | undefined {
  // Node / Jest / browser with process.env shim: read from process.env
  // when available. This is always safe in TypeScript regardless of
  // module target, unlike import.meta.
  if (typeof process !== 'undefined' && (process as any).env) {
    const value = (process as any).env[name];
    if (typeof value === 'string') {
      return value;
    }
  }

  return undefined;
}

export function flagEnabled(name: string): boolean {
  const raw = readEnv(name);
  if (!raw) return false;
  return raw === '1' || raw === 'true' || raw === 'TRUE';
}

export function isSandboxAiStallDiagnosticsEnabled(): boolean {
  return flagEnabled('RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS');
}

export function isSandboxCaptureDebugEnabled(): boolean {
  return flagEnabled('RINGRIFT_SANDBOX_CAPTURE_DEBUG');
}

export function isSandboxAiCaptureDebugEnabled(): boolean {
  return flagEnabled('RINGRIFT_SANDBOX_AI_CAPTURE_DEBUG');
}

export function isSandboxAiTraceModeEnabled(): boolean {
  return flagEnabled('RINGRIFT_SANDBOX_AI_TRACE_MODE');
}

export function isSandboxAiParityModeEnabled(): boolean {
  return flagEnabled('RINGRIFT_SANDBOX_AI_PARITY_MODE');
}

/**
 * Global rules-backend mode selector.
 *
 * RINGRIFT_RULES_MODE:
 *   - 'ts'     : TypeScript backend rules are authoritative
 *   - 'python' : Python AI-service rules are authoritative
 *   - 'shadow' : TS authoritative, Python evaluated in shadow for parity
 */
export type RulesMode = 'ts' | 'python' | 'shadow';

/**
 * Read the current rules-backend mode from the environment.
 * Defaults to 'ts' when unset or invalid.
 */
export function getRulesMode(): RulesMode {
  const raw = readEnv('RINGRIFT_RULES_MODE');
  if (raw === 'python' || raw === 'shadow') {
    return raw;
  }
  return 'ts';
}

/** True when running with Python rules authoritative. */
export function isPythonRulesMode(): boolean {
  return getRulesMode() === 'python';
}

/** True when running in TS-authoritative, Python-shadow mode. */
export function isRulesShadowMode(): boolean {
  return getRulesMode() === 'shadow';
}
