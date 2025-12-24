/**
 * @fileoverview Sandbox Rules-UX Telemetry - UTILITY, NOT CANONICAL
 *
 * SSoT alignment: This module provides **telemetry utilities** for sandbox.
 * It contains no rules logic.
 *
 * This utility:
 * - Emits spec-aligned sandbox_* telemetry events
 * - Only curated learning scenarios emit events
 * - High-cardinality identifiers stay in payload, not labels
 *
 * DO NOT add rules logic here - this is a pure telemetry utility module.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 * @see UX_RULES_TELEMETRY_SPEC.md
 */

import type { LoadableScenario } from './scenarioTypes';
import type { GameResult } from '../../shared/types/game';
import { logRulesUxEvent } from '../utils/rulesUxTelemetry';
import { getRulesUxContextForScenarioId } from '../../shared/teaching/scenarioTelemetry';

/**
 * Emit a sandbox_scenario_loaded event when a curated scenario is loaded
 * into the sandbox host.
 *
 * This should be called when the scenario is actually initialised into the
 * sandbox engine, not merely when metadata is fetched.
 */
export async function logSandboxScenarioLoaded(scenario: LoadableScenario): Promise<void> {
  // Only curated learning scenarios participate in rules-UX sandbox telemetry.
  if (scenario.source !== 'curated') return;

  const rulesContext = getRulesUxContextForScenarioId(scenario.id);

  await logRulesUxEvent({
    type: 'sandbox_scenario_loaded',
    source: 'sandbox',
    boardType: scenario.boardType,
    numPlayers: scenario.playerCount,
    rulesContext,
    rulesConcept: scenario.rulesConcept,
    scenarioId: scenario.id,
    isSandbox: true,
    payload: {
      origin: 'sandbox_browser',
    },
  });
}

/**
 * Emit a sandbox_scenario_completed event when play through a curated
 * sandbox scenario reaches a terminal game state.
 *
 * The current implementation treats any non-abandonment terminal reason as
 * a "successful" completion of the scenario; future passes can refine this
 * based on per-scenario objectives.
 */
export async function logSandboxScenarioCompleted(args: {
  scenario: LoadableScenario;
  victoryReason?: GameResult['reason'] | null;
}): Promise<void> {
  const { scenario, victoryReason } = args;

  // Only curated learning scenarios participate in rules-UX sandbox telemetry.
  if (scenario.source !== 'curated') return;

  const rulesContext = getRulesUxContextForScenarioId(scenario.id);

  const success =
    victoryReason === 'ring_elimination' ||
    victoryReason === 'territory_control' ||
    victoryReason === 'last_player_standing';

  await logRulesUxEvent({
    type: 'sandbox_scenario_completed',
    source: 'sandbox',
    boardType: scenario.boardType,
    numPlayers: scenario.playerCount,
    rulesContext,
    rulesConcept: scenario.rulesConcept,
    scenarioId: scenario.id,
    isSandbox: true,
    payload: {
      success,
      victoryReason: victoryReason ?? 'unknown',
    },
  });
}
