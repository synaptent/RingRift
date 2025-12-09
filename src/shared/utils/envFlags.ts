// Shared helpers for reading environment flags in both Node/Jest and the
// browser bundle. For browser builds, this relies on the bundler exposing
// process.env-style variables (for example via Vite define/plugin wiring).
// Keeping this logic centralised ensures sandbox AI diagnostics behave
// consistently across environments.

// Type-safe process.env access that works in both Node and browser contexts
type ProcessEnv = Record<string, string | undefined>;
function getProcessEnv(): ProcessEnv | undefined {
  if (typeof process !== 'undefined' && typeof process.env === 'object') {
    return process.env as ProcessEnv;
  }
  return undefined;
}

export function readEnv(name: string): string | undefined {
  // Node / Jest / browser with process.env shim: read from process.env
  // when available. This is always safe in TypeScript regardless of
  // module target, unlike import.meta.
  const env = getProcessEnv();
  if (env) {
    const value = env[name];
    if (typeof value === 'string') {
      return value;
    }
  }

  return undefined;
}

/**
 * Returns true if running in a test environment (NODE_ENV === 'test').
 * Use this instead of manually checking process.env for consistent behavior.
 */
export function isTestEnvironment(): boolean {
  return readEnv('NODE_ENV') === 'test';
}

/**
 * Returns true if running inside a Jest worker process.
 * This is useful for detecting test runtime even when NODE_ENV might be
 * configured differently (e.g., NODE_ENV=development in Jest).
 */
export function isJestRuntime(): boolean {
  return readEnv('JEST_WORKER_ID') !== undefined;
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

/**
 * Debug flag for sandbox Last-Player-Standing (LPS) and early game-end
 * behaviour. When enabled, the sandbox engine emits additional console
 * diagnostics around LPS round tracking, real-action detection, and
 * victory evaluation so reproducible early-completion bugs can be
 * investigated from the browser console.
 */
export function isSandboxLpsDebugEnabled(): boolean {
  return flagEnabled('RINGRIFT_SANDBOX_LPS_DEBUG');
}

export function isSandboxAiParityModeEnabled(): boolean {
  return flagEnabled('RINGRIFT_SANDBOX_AI_PARITY_MODE');
}

/**
 * Debug flag for visual/behavioural sandbox board animations (movement,
 * captures, destination pulses). When enabled, client-side code may emit
 * additional console diagnostics describing how animations were derived
 * from GameState and board diffs.
 */
export function isSandboxAnimationDebugEnabled(): boolean {
  return flagEnabled('RINGRIFT_SANDBOX_ANIMATION_DEBUG');
}

/**
 * Feature flag for enabling experimental local heuristic-based AI selection
 * in TypeScript hosts (backend fallback + sandbox AI).
 *
 * When enabled, callers may choose to invoke the shared heuristic evaluator
 * for diagnostics or tie-breaking, while keeping the existing bucketed
 * selection policy as the default. This flag is intentionally decoupled
 * from parity-mode flags so that heuristic behaviour can be explored in
 * isolation without affecting RNG-parity or Python-aligned test suites.
 */
export function isLocalAIHeuristicModeEnabled(): boolean {
  return flagEnabled('RINGRIFT_LOCAL_AI_HEURISTIC_MODE');
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

/**
 * FSM validation mode selector.
 *
 * RINGRIFT_FSM_VALIDATION_MODE:
 *   - 'off'    : No FSM validation (legacy mode)
 *   - 'shadow' : FSM runs in parallel, logs divergences without affecting behavior
 *   - 'active' : FSM validation is authoritative, rejects invalid moves
 *
 * For backwards compatibility, RINGRIFT_FSM_SHADOW_VALIDATION=1 is equivalent
 * to setting mode='shadow'.
 */
export type FSMValidationMode = 'off' | 'shadow' | 'active';

/**
 * Read the current FSM validation mode from the environment.
 * Defaults to 'off' when unset or invalid.
 *
 * For backwards compatibility:
 * - RINGRIFT_FSM_SHADOW_VALIDATION=1 implies 'shadow' mode
 * - RINGRIFT_FSM_VALIDATION_MODE takes precedence if set
 */
export function getFSMValidationMode(): FSMValidationMode {
  // Explicit mode takes precedence
  const modeRaw = readEnv('RINGRIFT_FSM_VALIDATION_MODE');
  if (modeRaw === 'shadow' || modeRaw === 'active' || modeRaw === 'off') {
    return modeRaw;
  }

  // Backwards compat: legacy shadow flag
  if (flagEnabled('RINGRIFT_FSM_SHADOW_VALIDATION')) {
    return 'shadow';
  }

  return 'off';
}

/** True when FSM validation is in shadow mode (run but don't enforce). */
export function isFSMShadowMode(): boolean {
  return getFSMValidationMode() === 'shadow';
}

/** True when FSM validation is active and authoritative. */
export function isFSMActiveMode(): boolean {
  return getFSMValidationMode() === 'active';
}

/** True when any FSM validation is enabled (shadow or active). */
export function isFSMValidationEnabled(): boolean {
  const mode = getFSMValidationMode();
  return mode === 'shadow' || mode === 'active';
}

/**
 * True when FSM shadow orchestration checks are enabled.
 *
 * When enabled, the orchestrator will run the FSM transition logic in
 * parallel and log any divergences between FSM-derived phase/player and
 * the legacy orchestration result. This does not alter game behavior.
 */
export function isFSMOrchestratorShadowEnabled(): boolean {
  return flagEnabled('RINGRIFT_FSM_ORCHESTRATOR_SHADOW');
}

/**
 * Debug logging wrapper that suppresses ESLint no-console warnings.
 * Use this for debug logs that are gated by environment flags.
 * The wrapped console.log is only invoked if the condition is true.
 *
 * @example
 * debugLog(flagEnabled('RINGRIFT_DEBUG'), 'Debug message', data);
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function debugLog(condition: boolean, ...args: any[]): void {
  if (condition) {
    // eslint-disable-next-line no-console
    console.log(...args);
  }
}
