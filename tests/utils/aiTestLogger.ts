import fs from 'fs';
import path from 'path';

/**
 * Centralised logging helper for AI-heavy Jest suites (backend & sandbox).
 *
 * Goals:
 * - Keep default Jest output small so large AI simulations dont spam the
 *   terminal or blow out editor contexts.
 * - Still capture rich diagnostic snapshots (including full board state)
 *   when something goes wrong.
 * - Allow developers to opt into verbose console logging via an env var.
 *
 * Behaviour:
 * - By default, detailed diagnostics are appended as line-delimited JSON
 *   to `logs/ai/<stream>.log` and Jest only sees the normal assertion
 *   failures / Error messages.
 * - When RINGRIFT_AI_DEBUG=1 (or `true`), diagnostics are ALSO emitted to
 *   the console via console.error so that local debugging behaves as
 *   before.
 */

const AI_LOG_DIR = path.join(process.cwd(), 'logs', 'ai');

function ensureLogDir(): void {
  try {
    fs.mkdirSync(AI_LOG_DIR, { recursive: true });
  } catch {
    // Best-effort only; if this fails we fall back to console logging.
  }
}

const DEBUG_ENABLED =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_AI_DEBUG ?? '');

let hintedOnce = false;

export type AiDiagnosticStream = string;

export function logAiDiagnostic(
  label: string,
  payload: unknown,
  stream: AiDiagnosticStream = 'generic'
): void {
  const entry = {
    timestamp: new Date().toISOString(),
    stream,
    label,
    payload
  };

  // First, try to persist the full payload to a log file so developers can
  // inspect it even when console output is kept minimal.
  try {
    ensureLogDir();
    const filePath = path.join(AI_LOG_DIR, `${stream}.log`);
    fs.appendFileSync(filePath, JSON.stringify(entry) + '\n', { encoding: 'utf8' });
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error('[AI-Tests] Failed to write diagnostic log file', err);
  }

  if (DEBUG_ENABLED) {
    // In explicit debug mode, mirror the full diagnostic to the console.
    // eslint-disable-next-line no-console
    console.error(`[AI-Tests:${stream}] ${label}`, payload);
  } else if (!hintedOnce) {
    // In normal test runs, emit a single one-line hint so that when a
    // failure occurs developers know where to look for rich diagnostics
    // and how to re-run with verbose console output.
    hintedOnce = true;
    // eslint-disable-next-line no-console
    console.log(
      '[AI-Tests] Detailed diagnostics are being written to logs/ai/*.log. ' +
        'Set RINGRIFT_AI_DEBUG=1 to enable verbose console output.'
    );
  }
}
