/**
 * ShadowModeComparator
 *
 * Runs both legacy and orchestrator engines in parallel during shadow mode rollout,
 * compares results, logs discrepancies, and tracks metrics for validation.
 *
 * Key behaviors:
 * - Always returns the legacy result (zero risk)
 * - Compares key game state fields between engines
 * - Logs mismatches at warn level for analysis
 * - Tracks orchestrator errors gracefully
 *
 * @see docs/drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md
 */

import { logger } from '../utils/logger';
import type { GameState, GameResult, RingStack } from '../../shared/types/game';

/**
 * Result of applying a move through an engine.
 * This matches the return type of GameEngine.makeMove().
 */
export interface MoveResult {
  success: boolean;
  error?: string | undefined;
  gameState?: GameState | undefined;
  gameResult?: GameResult | undefined;
}

/**
 * Comparison result from shadow mode execution.
 */
export interface ShadowComparison {
  sessionId: string;
  moveNumber: number;
  legacyResult: MoveResult;
  orchestratorResult: MoveResult | null;
  isMatch: boolean;
  differences: string[];
  legacyLatencyMs: number;
  orchestratorLatencyMs: number;
}

/**
 * Aggregated metrics from shadow mode comparisons.
 */
export interface ShadowMetrics {
  totalComparisons: number;
  matches: number;
  mismatches: number;
  mismatchRate: number;
  orchestratorErrors: number;
  orchestratorErrorRate: number;
  avgLegacyLatencyMs: number;
  avgOrchestratorLatencyMs: number;
}

/**
 * ShadowModeComparator runs both engines in parallel and compares results.
 */
export class ShadowModeComparator {
  private comparisons: ShadowComparison[] = [];
  private readonly maxStoredComparisons: number;
  private orchestratorErrorCount = 0;

  constructor(maxStoredComparisons = 1000) {
    this.maxStoredComparisons = maxStoredComparisons;
  }

  /**
   * Run both engines in parallel and compare results.
   * @returns The legacy result (safe output) plus comparison data
   */
  async compare(
    sessionId: string,
    moveNumber: number,
    legacyEngine: () => Promise<MoveResult>,
    orchestratorEngine: () => Promise<MoveResult>
  ): Promise<{ result: MoveResult; comparison: ShadowComparison }> {
    const startTime = Date.now();

    // Run both engines in parallel
    const [legacyOutcome, orchestratorOutcome] = await Promise.all([
      this.runWithTiming(legacyEngine, startTime),
      this.runOrchestratorSafe(orchestratorEngine, sessionId, startTime),
    ]);

    const comparison = this.createComparison(
      sessionId,
      moveNumber,
      legacyOutcome.result,
      orchestratorOutcome?.result ?? null,
      legacyOutcome.latency,
      orchestratorOutcome?.latency ?? -1
    );

    this.storeComparison(comparison);
    this.logComparison(comparison);

    // Always return legacy result (zero risk)
    return { result: legacyOutcome.result, comparison };
  }

  /**
   * Run an engine function and capture timing.
   */
  private async runWithTiming(
    engine: () => Promise<MoveResult>,
    startTime: number
  ): Promise<{ result: MoveResult; latency: number }> {
    const result = await engine();
    return { result, latency: Date.now() - startTime };
  }

  /**
   * Run orchestrator engine safely, catching any errors.
   */
  private async runOrchestratorSafe(
    engine: () => Promise<MoveResult>,
    sessionId: string,
    startTime: number
  ): Promise<{ result: MoveResult; latency: number } | null> {
    try {
      const result = await engine();
      return { result, latency: Date.now() - startTime };
    } catch (err) {
      this.orchestratorErrorCount++;
      logger.error('Shadow mode: orchestrator engine error', {
        err,
        sessionId,
        errorMessage: err instanceof Error ? err.message : String(err),
      });
      return null;
    }
  }

  /**
   * Compare two MoveResults and identify differences.
   */
  private createComparison(
    sessionId: string,
    moveNumber: number,
    legacy: MoveResult,
    orchestrator: MoveResult | null,
    legacyLatency: number,
    orchestratorLatency: number
  ): ShadowComparison {
    const differences: string[] = [];

    if (!orchestrator) {
      differences.push('orchestrator_error');
    } else {
      // Compare success status
      if (legacy.success !== orchestrator.success) {
        differences.push(`success: ${legacy.success} vs ${orchestrator.success}`);
      }

      // Compare error messages
      if (legacy.error !== orchestrator.error) {
        differences.push(`error: "${legacy.error ?? 'none'}" vs "${orchestrator.error ?? 'none'}"`);
      }

      // Compare game state if both have it
      if (legacy.gameState && orchestrator.gameState) {
        this.compareGameStates(legacy.gameState, orchestrator.gameState, differences);
      } else if (legacy.gameState && !orchestrator.gameState) {
        differences.push('gameState: legacy has state, orchestrator missing');
      } else if (!legacy.gameState && orchestrator.gameState) {
        differences.push('gameState: legacy missing, orchestrator has state');
      }

      // Compare game result if present
      if (legacy.gameResult || orchestrator.gameResult) {
        this.compareGameResults(legacy.gameResult, orchestrator.gameResult, differences);
      }
    }

    return {
      sessionId,
      moveNumber,
      legacyResult: legacy,
      orchestratorResult: orchestrator!,
      isMatch: differences.length === 0,
      differences,
      legacyLatencyMs: legacyLatency,
      orchestratorLatencyMs: orchestratorLatency,
    };
  }

  /**
   * Compare two GameState objects and add differences to the list.
   */
  private compareGameStates(
    legacy: GameState,
    orchestrator: GameState,
    differences: string[]
  ): void {
    // Core game state fields
    if (legacy.currentPlayer !== orchestrator.currentPlayer) {
      differences.push(`currentPlayer: ${legacy.currentPlayer} vs ${orchestrator.currentPlayer}`);
    }

    if (legacy.currentPhase !== orchestrator.currentPhase) {
      differences.push(`phase: ${legacy.currentPhase} vs ${orchestrator.currentPhase}`);
    }

    if (legacy.gameStatus !== orchestrator.gameStatus) {
      differences.push(`gameStatus: ${legacy.gameStatus} vs ${orchestrator.gameStatus}`);
    }

    if (legacy.winner !== orchestrator.winner) {
      differences.push(`winner: ${legacy.winner ?? 'none'} vs ${orchestrator.winner ?? 'none'}`);
    }

    // Turn number (move history length)
    const legacyTurnNumber = legacy.moveHistory?.length ?? 0;
    const orchestratorTurnNumber = orchestrator.moveHistory?.length ?? 0;
    if (legacyTurnNumber !== orchestratorTurnNumber) {
      differences.push(`turnNumber: ${legacyTurnNumber} vs ${orchestratorTurnNumber}`);
    }

    // Ring counts
    if (legacy.totalRingsInPlay !== orchestrator.totalRingsInPlay) {
      differences.push(
        `totalRingsInPlay: ${legacy.totalRingsInPlay} vs ${orchestrator.totalRingsInPlay}`
      );
    }

    if (legacy.totalRingsEliminated !== orchestrator.totalRingsEliminated) {
      differences.push(
        `totalRingsEliminated: ${legacy.totalRingsEliminated} vs ${orchestrator.totalRingsEliminated}`
      );
    }

    // Board state comparisons
    this.compareBoardStates(legacy, orchestrator, differences);

    // Player state comparisons
    this.comparePlayerStates(legacy, orchestrator, differences);
  }

  /**
   * Compare board states (stacks, markers, collapsed spaces, territories).
   */
  private compareBoardStates(
    legacy: GameState,
    orchestrator: GameState,
    differences: string[]
  ): void {
    const legacyBoard = legacy.board;
    const orchBoard = orchestrator.board;

    // Compare stacks (board positions)
    const legacyStackKeys = new Set(legacyBoard.stacks.keys());
    const orchStackKeys = new Set(orchBoard.stacks.keys());

    // Check for missing/extra stacks
    for (const key of legacyStackKeys) {
      if (!orchStackKeys.has(key)) {
        differences.push(`stack at ${key}: exists in legacy, missing in orchestrator`);
      }
    }
    for (const key of orchStackKeys) {
      if (!legacyStackKeys.has(key)) {
        differences.push(`stack at ${key}: missing in legacy, exists in orchestrator`);
      }
    }

    // Compare stack contents at common positions
    for (const key of legacyStackKeys) {
      if (orchStackKeys.has(key)) {
        const legacyStack = legacyBoard.stacks.get(key)!;
        const orchStack = orchBoard.stacks.get(key)!;
        this.compareStacks(key, legacyStack, orchStack, differences);
      }
    }

    // Compare markers
    const legacyMarkerKeys = new Set(legacyBoard.markers.keys());
    const orchMarkerKeys = new Set(orchBoard.markers.keys());

    if (legacyMarkerKeys.size !== orchMarkerKeys.size) {
      differences.push(`markerCount: ${legacyMarkerKeys.size} vs ${orchMarkerKeys.size}`);
    }

    // Compare collapsed spaces (territory)
    const legacyCollapsedKeys = new Set(legacyBoard.collapsedSpaces.keys());
    const orchCollapsedKeys = new Set(orchBoard.collapsedSpaces.keys());

    if (legacyCollapsedKeys.size !== orchCollapsedKeys.size) {
      differences.push(
        `collapsedSpacesCount: ${legacyCollapsedKeys.size} vs ${orchCollapsedKeys.size}`
      );
    }

    // Compare formed lines
    const legacyLineCount = legacyBoard.formedLines?.length ?? 0;
    const orchLineCount = orchBoard.formedLines?.length ?? 0;
    if (legacyLineCount !== orchLineCount) {
      differences.push(`formedLinesCount: ${legacyLineCount} vs ${orchLineCount}`);
    }

    // Compare eliminated rings per player
    const legacyEliminated = legacyBoard.eliminatedRings ?? {};
    const orchEliminated = orchBoard.eliminatedRings ?? {};
    const allElimPlayers = new Set([
      ...Object.keys(legacyEliminated),
      ...Object.keys(orchEliminated),
    ]);

    for (const player of allElimPlayers) {
      const legacyCount = legacyEliminated[Number(player)] ?? 0;
      const orchCount = orchEliminated[Number(player)] ?? 0;
      if (legacyCount !== orchCount) {
        differences.push(`eliminatedRings[${player}]: ${legacyCount} vs ${orchCount}`);
      }
    }
  }

  /**
   * Compare two ring stacks.
   */
  private compareStacks(
    key: string,
    legacy: RingStack,
    orchestrator: RingStack,
    differences: string[]
  ): void {
    if (legacy.stackHeight !== orchestrator.stackHeight) {
      differences.push(
        `stack[${key}].height: ${legacy.stackHeight} vs ${orchestrator.stackHeight}`
      );
    }

    if (legacy.capHeight !== orchestrator.capHeight) {
      differences.push(`stack[${key}].capHeight: ${legacy.capHeight} vs ${orchestrator.capHeight}`);
    }

    if (legacy.controllingPlayer !== orchestrator.controllingPlayer) {
      differences.push(
        `stack[${key}].controller: ${legacy.controllingPlayer} vs ${orchestrator.controllingPlayer}`
      );
    }

    // Compare ring composition
    const legacyRings = legacy.rings.join(',');
    const orchRings = orchestrator.rings.join(',');
    if (legacyRings !== orchRings) {
      differences.push(`stack[${key}].rings: [${legacyRings}] vs [${orchRings}]`);
    }
  }

  /**
   * Compare player states (scores, rings in hand, territory).
   */
  private comparePlayerStates(
    legacy: GameState,
    orchestrator: GameState,
    differences: string[]
  ): void {
    const legacyPlayers = legacy.players ?? [];
    const orchPlayers = orchestrator.players ?? [];

    if (legacyPlayers.length !== orchPlayers.length) {
      differences.push(`playerCount: ${legacyPlayers.length} vs ${orchPlayers.length}`);
      return;
    }

    for (let i = 0; i < legacyPlayers.length; i++) {
      const lp = legacyPlayers[i];
      const op = orchPlayers[i];

      if (!lp || !op) continue;

      if (lp.ringsInHand !== op.ringsInHand) {
        differences.push(
          `player[${lp.playerNumber}].ringsInHand: ${lp.ringsInHand} vs ${op.ringsInHand}`
        );
      }

      if (lp.eliminatedRings !== op.eliminatedRings) {
        differences.push(
          `player[${lp.playerNumber}].eliminatedRings: ${lp.eliminatedRings} vs ${op.eliminatedRings}`
        );
      }

      if (lp.territorySpaces !== op.territorySpaces) {
        differences.push(
          `player[${lp.playerNumber}].territorySpaces: ${lp.territorySpaces} vs ${op.territorySpaces}`
        );
      }
    }
  }

  /**
   * Compare game results.
   */
  private compareGameResults(
    legacy: MoveResult['gameResult'],
    orchestrator: MoveResult['gameResult'],
    differences: string[]
  ): void {
    if (!legacy && !orchestrator) return;

    if (!legacy) {
      differences.push('gameResult: missing in legacy, present in orchestrator');
      return;
    }

    if (!orchestrator) {
      differences.push('gameResult: present in legacy, missing in orchestrator');
      return;
    }

    if (legacy.winner !== orchestrator.winner) {
      differences.push(
        `gameResult.winner: ${legacy.winner ?? 'none'} vs ${orchestrator.winner ?? 'none'}`
      );
    }

    if (legacy.reason !== orchestrator.reason) {
      differences.push(`gameResult.reason: ${legacy.reason} vs ${orchestrator.reason}`);
    }
  }

  /**
   * Log comparison with appropriate level.
   */
  private logComparison(comparison: ShadowComparison): void {
    if (comparison.isMatch) {
      logger.debug('Shadow mode: engines match', {
        sessionId: comparison.sessionId,
        moveNumber: comparison.moveNumber,
        legacyLatencyMs: comparison.legacyLatencyMs,
        orchestratorLatencyMs: comparison.orchestratorLatencyMs,
      });
    } else {
      logger.warn('Shadow mode: ENGINE MISMATCH', {
        sessionId: comparison.sessionId,
        moveNumber: comparison.moveNumber,
        differences: comparison.differences,
        differenceCount: comparison.differences.length,
        legacyLatencyMs: comparison.legacyLatencyMs,
        orchestratorLatencyMs: comparison.orchestratorLatencyMs,
      });
    }
  }

  /**
   * Store comparison with limit.
   */
  private storeComparison(comparison: ShadowComparison): void {
    this.comparisons.push(comparison);
    if (this.comparisons.length > this.maxStoredComparisons) {
      this.comparisons.shift(); // Remove oldest
    }
  }

  /**
   * Get metrics for monitoring.
   */
  getMetrics(): ShadowMetrics {
    const total = this.comparisons.length;
    const matches = this.comparisons.filter((c) => c.isMatch).length;
    const mismatches = total - matches;

    // Calculate average latencies
    const avgLegacy =
      total > 0 ? this.comparisons.reduce((sum, c) => sum + c.legacyLatencyMs, 0) / total : 0;

    const validOrchestratorComparisons = this.comparisons.filter(
      (c) => c.orchestratorLatencyMs >= 0
    );
    const avgOrchestrator =
      validOrchestratorComparisons.length > 0
        ? validOrchestratorComparisons.reduce((sum, c) => sum + c.orchestratorLatencyMs, 0) /
          validOrchestratorComparisons.length
        : 0;

    const orchestratorErrors = this.comparisons.filter((c) =>
      c.differences.includes('orchestrator_error')
    ).length;

    return {
      totalComparisons: total,
      matches,
      mismatches,
      mismatchRate: total > 0 ? mismatches / total : 0,
      orchestratorErrors,
      orchestratorErrorRate: total > 0 ? orchestratorErrors / total : 0,
      avgLegacyLatencyMs: Math.round(avgLegacy * 100) / 100,
      avgOrchestratorLatencyMs: Math.round(avgOrchestrator * 100) / 100,
    };
  }

  /**
   * Get recent mismatches for analysis.
   */
  getRecentMismatches(limit = 10): ShadowComparison[] {
    return this.comparisons.filter((c) => !c.isMatch).slice(-limit);
  }

  /**
   * Get all stored comparisons.
   */
  getAllComparisons(): ShadowComparison[] {
    return [...this.comparisons];
  }

  /**
   * Clear stored comparisons.
   */
  clearComparisons(): void {
    this.comparisons = [];
    this.orchestratorErrorCount = 0;
  }

  /**
   * Get the total number of orchestrator errors tracked.
   */
  getOrchestratorErrorCount(): number {
    return this.orchestratorErrorCount;
  }
}

// Singleton instance for easy import
export const shadowComparator = new ShadowModeComparator();
