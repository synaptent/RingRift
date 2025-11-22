import client from 'prom-client';
import { logger } from './logger';

/**
 * Prometheus counters for TS <-> Python rules-engine parity.
 *
 * These are incremented by the RulesBackendFacade whenever it observes
 * a discrepancy between the authoritative backend engine and the Python
 * rules service running in shadow mode (or vice versa when Python is
 * authoritative).
 */
export const rulesParityMetrics = {
  validMismatch: new client.Counter({
    name: 'rules_parity_valid_mismatch_total',
    help: 'TS vs Python rules: validation verdict mismatch count',
  }),
  hashMismatch: new client.Counter({
    name: 'rules_parity_hash_mismatch_total',
    help: 'TS vs Python rules: post-move state hash mismatch count',
  }),
  sMismatch: new client.Counter({
    name: 'rules_parity_S_mismatch_total',
    help: 'TS vs Python rules: S-invariant mismatch count',
  }),
  gameStatusMismatch: new client.Counter({
    name: 'rules_parity_gameStatus_mismatch_total',
    help: 'TS vs Python rules: gameStatus mismatch count',
  }),
};

/**
 * Structured logging helper for rules parity discrepancies.
 *
 * All parity-related logs are emitted with a common message key so that
 * they can be discovered and aggregated easily in log pipelines.
 */
export function logRulesMismatch(
  kind: 'valid' | 'hash' | 'S' | 'gameStatus' | 'backend_fallback' | 'shadow_error',
  details: Record<string, unknown>
): void {
  logger.warn('rules_parity_mismatch', {
    kind,
    ...details,
  });
}
