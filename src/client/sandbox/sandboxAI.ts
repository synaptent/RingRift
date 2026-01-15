/**
 * @fileoverview Sandbox AI Turn Logic - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This module is an **adapter** over the canonical shared engine.
 * It coordinates AI move selection and application for sandbox/offline games.
 *
 * Canonical SSoT:
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - Move selection: `src/shared/engine/localMoveSelector.ts`
 * - Types: `src/shared/types/game.ts`
 *
 * This adapter:
 * - Orchestrates AI turn execution using hooks to ClientSandboxEngine
 * - Delegates move enumeration to canonical shared helpers
 * - Uses service-backed AI when available, falls back to local heuristics
 * - Tracks AI stalls and diagnostics for debugging
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 * AI move selection and evaluation belong in the shared layer or ai-service.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import type {
  GameState,
  Move,
  Position,
  RingStack,
  BoardState,
  LocalAIRng,
} from '../../shared/engine';
import {
  positionToString,
  BOARD_CONFIGS,
  hashGameState,
  chooseLocalMoveFromCandidates,
  evaluateSkipPlacementEligibility as evaluateSkipPlacementEligibilityAggregate,
  isANMState,
  computeGlobalLegalActionsSummary,
  evaluateVictory,
} from '../../shared/engine';
import { serializeGameState } from '../../shared/engine/contracts/serialization';
import {
  isSandboxAiCaptureDebugEnabled,
  isSandboxAiStallDiagnosticsEnabled,
  isSandboxAiParityModeEnabled,
} from '../../shared/utils/envFlags';
import { normalizeLegacyMoveType } from '../../shared/engine/legacy/legacyMoveTypes';
import { recordSandboxAiDiagnostics, type SandboxAiDecisionSource } from './sandboxAiDiagnostics';
import { getSandboxAIServiceAvailable } from '../utils/aiServiceAvailability';

const SANDBOX_AI_CAPTURE_DEBUG_ENABLED = isSandboxAiCaptureDebugEnabled();
const SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED = isSandboxAiStallDiagnosticsEnabled();

/** Type-safe window extension for sandbox AI diagnostic traces. */
interface SandboxTraceWindow extends Window {
  __RINGRIFT_SANDBOX_TRACE__?: SandboxAITurnTraceEntry[];
}

/**
 * Canonical sandbox stall threshold for consecutive no-op AI turns.
 *
 * This is intentionally aligned with the test-layer STALL_WINDOW_STEPS
 * defined in tests/utils/aiSimulationPolicy.ts (currently 8) so that:
 *   - in-browser 'stall' trace entries,
 *   - sandbox fuzz harness classification,
 *   - and backend/sandbox parity diagnostics
 * all share the same semantic window for structural stalls.
 */
const SANDBOX_NOOP_STALL_THRESHOLD = 8;
const SANDBOX_NOOP_MAX_THRESHOLD = 10; // Stop execution after this many consecutive no-ops
const MAX_SANDBOX_TRACE_ENTRIES = 2000;

/**
 * Exported view of the sandbox stall window so tests and diagnostic harnesses
 * can assert alignment with the canonical STALL_WINDOW_STEPS used by
 * aiSimulationPolicy without reaching into internal constants.
 */
export const SANDBOX_STALL_WINDOW_STEPS = SANDBOX_NOOP_STALL_THRESHOLD;

/**
 * Apply the stalemate ladder tie-breaker to determine a winner when the AI
 * stalls but evaluateVictory() returns isGameOver=false. This mirrors the
 * ladder logic from VictoryAggregate.ts but is used as a fallback for sandbox
 * stall detection when standard victory conditions haven't been met.
 *
 * Tie-break order:
 * 1. Territory spaces (most wins)
 * 2. Eliminated rings (most wins)
 * 3. Markers on board (most wins)
 * 4. Last actor (player who moved before current player)
 */
function applyStalemateLadder(state: GameState): number | undefined {
  const players = state.players;
  if (!players || players.length === 0) {
    return undefined;
  }

  // 1. Territory spaces
  const maxTerritory = Math.max(...players.map((p) => p.territorySpaces));
  const territoryLeaders = players.filter((p) => p.territorySpaces === maxTerritory);
  if (territoryLeaders.length === 1 && maxTerritory > 0) {
    return territoryLeaders[0].playerNumber;
  }

  // 2. Eliminated rings
  const maxEliminated = Math.max(...players.map((p) => p.eliminatedRings));
  const eliminationLeaders = players.filter((p) => p.eliminatedRings === maxEliminated);
  if (eliminationLeaders.length === 1 && maxEliminated > 0) {
    return eliminationLeaders[0].playerNumber;
  }

  // 3. Markers on board
  const markerCountsByPlayer: { [player: number]: number } = {};
  for (const p of players) {
    markerCountsByPlayer[p.playerNumber] = 0;
  }
  for (const marker of state.board.markers.values()) {
    const owner = marker.player;
    if (markerCountsByPlayer[owner] !== undefined) {
      markerCountsByPlayer[owner] += 1;
    }
  }
  const maxMarkers = Math.max(...players.map((p) => markerCountsByPlayer[p.playerNumber] ?? 0));
  const markerLeaders = players.filter(
    (p) => (markerCountsByPlayer[p.playerNumber] ?? 0) === maxMarkers
  );
  if (markerLeaders.length === 1 && maxMarkers > 0) {
    return markerLeaders[0].playerNumber;
  }

  // 4. Last actor (player immediately preceding currentPlayer in turn order)
  const currentIdx = players.findIndex((p) => p.playerNumber === state.currentPlayer);
  if (currentIdx !== -1) {
    const lastIdx = (currentIdx - 1 + players.length) % players.length;
    return players[lastIdx].playerNumber;
  }

  // Fallback: first player
  return players[0]?.playerNumber;
}

// Module-level counter tracking consecutive AI turns that leave the
// sandbox GameState hash unchanged while the same AI player remains
// to move. Used as a structural stall detector in dev/test builds.
let sandboxConsecutiveNoopAITurns = 0;
let sandboxStallLoggingSuppressed = false;

function clampDifficulty(raw: number | undefined, fallback: number): number {
  const value = typeof raw === 'number' && Number.isFinite(raw) ? raw : fallback;
  return Math.max(1, Math.min(10, Math.round(value)));
}

function positionKey(pos: Position | undefined): string {
  if (!pos) return '';
  const z = pos.z !== undefined ? String(pos.z) : '';
  return `${pos.x},${pos.y},${z}`;
}

function positionsKey(positions: Position[] | undefined): string {
  if (!positions || positions.length === 0) return '';
  return positions.map((p) => positionKey(p)).join('|');
}

function territoriesKey(territories: Move['disconnectedRegions'] | undefined): string {
  if (!territories || territories.length === 0) return '';
  return territories
    .map((t) => `${t.controllingPlayer}:${t.isDisconnected ? 'd' : 'c'}:${positionsKey(t.spaces)}`)
    .join('||');
}

function formedLinesKey(lines: Move['formedLines'] | undefined): string {
  if (!lines || lines.length === 0) return '';
  return lines.map((l) => `${l.player}:${positionsKey(l.positions)}`).join('||');
}

function moveMatchKey(move: Move): string {
  return [
    move.type,
    String(move.player),
    positionKey(move.from),
    positionKey(move.to),
    positionKey(move.captureTarget),
    String(move.buildAmount ?? ''),
    String(move.placementCount ?? ''),
    String(move.recoveryOption ?? ''),
    positionsKey(move.collapsedMarkers),
    formedLinesKey(move.formedLines),
    territoriesKey(move.disconnectedRegions),
    move.eliminationContext ?? '',
    move.eliminationFromStack ? positionKey(move.eliminationFromStack.position) : '',
  ].join('|');
}

async function tryRequestSandboxAIMove(payload: {
  state: ReturnType<typeof serializeGameState>;
  difficulty: number;
  playerNumber: number;
}): Promise<{
  move: Move;
  evaluation?: unknown;
  thinkingTimeMs?: number | null;
  aiType?: string;
  difficulty?: number;
  heuristicProfileId?: string | null;
  useNeuralNet?: boolean | null;
  nnModelId?: string | null;
  nnCheckpoint?: string | null;
  nnueCheckpoint?: string | null;
} | null> {
  if (typeof fetch !== 'function') {
    return null;
  }

  // Skip API call in production without AI service configured
  if (!getSandboxAIServiceAvailable()) {
    return null;
  }

  try {
    const response = await fetch('/api/games/sandbox/ai/move', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      return null;
    }

    const raw = (await response.json()) as unknown;
    if (!raw || typeof raw !== 'object') {
      return null;
    }

    const data = raw as Record<string, unknown>;
    const move = data.move as Move | null | undefined;
    if (!move) {
      return null;
    }

    const aiType = typeof data.aiType === 'string' ? data.aiType : undefined;
    const difficulty = typeof data.difficulty === 'number' ? data.difficulty : undefined;

    const heuristicProfileId =
      typeof data.heuristicProfileId === 'string'
        ? data.heuristicProfileId
        : data.heuristicProfileId === null
          ? null
          : undefined;

    const useNeuralNet =
      typeof data.useNeuralNet === 'boolean'
        ? data.useNeuralNet
        : data.useNeuralNet === null
          ? null
          : undefined;

    const nnModelId =
      typeof data.nnModelId === 'string'
        ? data.nnModelId
        : data.nnModelId === null
          ? null
          : undefined;
    const nnCheckpoint =
      typeof data.nnCheckpoint === 'string'
        ? data.nnCheckpoint
        : data.nnCheckpoint === null
          ? null
          : undefined;
    const nnueCheckpoint =
      typeof data.nnueCheckpoint === 'string'
        ? data.nnueCheckpoint
        : data.nnueCheckpoint === null
          ? null
          : undefined;

    const thinkingTimeMs =
      typeof data.thinkingTimeMs === 'number'
        ? data.thinkingTimeMs
        : data.thinkingTimeMs === null
          ? null
          : undefined;

    return {
      move,
      evaluation: data.evaluation,
      thinkingTimeMs,
      aiType,
      difficulty,
      heuristicProfileId,
      useNeuralNet,
      nnModelId,
      nnCheckpoint,
      nnueCheckpoint,
    };
  } catch {
    return null;
  }
}

/**
 * Reset module-level sandbox AI stall counters. Used by tests to ensure
 * isolation between test cases that run multiple sandbox AI turns.
 */
export function resetSandboxAIStallCounters(): void {
  sandboxConsecutiveNoopAITurns = 0;
  sandboxStallLoggingSuppressed = false;
}

interface SandboxAITurnTraceEntry {
  kind: 'ai_turn' | 'stall';
  timestamp: number;
  boardType: GameState['boardType'];
  playerNumber: number | null;
  currentPhaseBefore: GameState['currentPhase'];
  currentPhaseAfter: GameState['currentPhase'];
  gameStatusBefore: GameState['gameStatus'];
  gameStatusAfter: GameState['gameStatus'];
  beforeHash: string;
  afterHash: string;
  lastAIMoveType: Move['type'] | null;
  lastAIMovePlayer: number | null;
  captureCount?: number | undefined;
  simpleMoveCount?: number | undefined;
  placementCandidateCount?: number | undefined;
  forcedEliminationAttempted?: boolean | undefined;
  forcedEliminationEliminated?: boolean | undefined;
  consecutiveNoopAITurns?: number | undefined;
  aiDecisionSource?: SandboxAiDecisionSource | undefined;
  aiDifficultyRequested?: number | undefined;
  serviceAiType?: string | undefined;
  serviceDifficulty?: number | undefined;
  heuristicProfileId?: string | null | undefined;
  useNeuralNet?: boolean | null | undefined;
  nnModelId?: string | null | undefined;
  nnCheckpoint?: string | null | undefined;
  nnueCheckpoint?: string | null | undefined;
  thinkingTimeMs?: number | null | undefined;
  serviceError?: string | undefined;
}

declare global {
  interface Window {
    __RINGRIFT_SANDBOX_TRACE__?: SandboxAITurnTraceEntry[];
  }
}

/**
 * Get (and lazily initialise) the in-browser sandbox AI trace buffer
 * used for debugging AI stalls. In non-browser builds this returns
 * null so the rest of the code can no-op.
 *
 * NOTE: The stall-diagnostics flag still guards additional console
 * logging and warning spam, but the trace buffer itself is now always
 * available so that "Copy AI trace" never returns an empty array in
 * normal dev runs.
 */
function getSandboxTraceBuffer(): SandboxAITurnTraceEntry[] | null {
  if (typeof window === 'undefined') {
    return null;
  }

  const traceWindow = window as SandboxTraceWindow;
  if (!Array.isArray(traceWindow.__RINGRIFT_SANDBOX_TRACE__)) {
    traceWindow.__RINGRIFT_SANDBOX_TRACE__ = [];
  }

  return traceWindow.__RINGRIFT_SANDBOX_TRACE__;
}

export interface SandboxAIHooks {
  getPlayerStacks(playerNumber: number, board: BoardState): RingStack[];
  hasAnyLegalMoveOrCaptureFrom(from: Position, playerNumber: number, board: BoardState): boolean;
  enumerateLegalRingPlacements(playerNumber: number): Position[];
  /**
   * Canonical host-level legal moves for the current player/phase.
   * Backed by ClientSandboxEngine.getValidMoves so movement/capture
   * decisions share the same surface as backend getValidMoves.
   */
  getValidMovesForCurrentPlayer(): Move[];
  /**
   * Pure helper used by the sandbox AI to mirror the backend
   * no-dead-placement check for multi-ring placements. This must
   * not mutate the passed-in board.
   */
  createHypotheticalBoardWithPlacement(
    board: BoardState,
    position: Position,
    playerNumber: number,
    count?: number
  ): BoardState;
  tryPlaceRings(position: Position, count: number): Promise<boolean>;
  enumerateCaptureSegmentsFrom(
    from: Position,
    playerNumber: number
  ): Array<{ from: Position; target: Position; landing: Position }>;
  enumerateSimpleMovementLandings(playerNumber: number): Array<{ fromKey: string; to: Position }>;
  maybeProcessForcedEliminationForCurrentPlayer(): boolean;
  handleMovementClick(position: Position): Promise<void>;
  appendHistoryEntry(before: GameState, action: Move): void;
  getGameState(): GameState;
  setGameState(state: GameState): void;
  setLastAIMove(move: Move | null): void;
  setSelectedStackKey(key: string | undefined): void;
  getMustMoveFromStackKey(): string | undefined;
  applyCanonicalMove(move: Move): Promise<void>;
  /**
   * Parity hook: true when the sandbox engine has recorded a pending
   * territory self-elimination for the current player in the current
   * territory_processing cycle. Mirrors the backend
   * GameEngine.pendingTerritorySelfElimination flag.
   */
  hasPendingTerritorySelfElimination(): boolean;
  /**
   * Parity hook: true when the sandbox engine has recorded a pending
   * line-reward elimination for the current player in the current
   * line_processing cycle. Mirrors the backend
   * GameEngine.pendingLineRewardElimination flag.
   */
  hasPendingLineRewardElimination(): boolean;
  /**
   * Pie-rule hooks: mirror the backend GameEngine.shouldOfferSwapSidesMetaMove
   * gate and applySwapSidesMove behaviour so sandbox AI can optionally
   * invoke swap_sides under the same one-time conditions as humans.
   */
  canCurrentPlayerSwapSides(): boolean;
  applySwapSidesForCurrentPlayer(): boolean;
  /**
   * Get the AI difficulty level (1-10) for a specific player.
   * Returns undefined if no difficulty is set (uses default heuristic).
   */
  getAIDifficulty?(playerNumber: number): number | undefined;
}

/**
 * Simple heuristic used by the sandbox AI to decide whether to invoke the
 * pie rule (swap_sides) when it is available to Player 2.
 *
 * Current policy:
 * - Only ever considers swap_sides on square8 boards.
 * - Treats a "strong" opening as Player 1 placing their first ring on one of
 *   the four central squares (3,3), (3,4), (4,3), (4,4).
 * - For other openings/boards, the AI declines the pie rule even if it is
 *   technically available.
 *
 * This keeps behaviour deterministic under a fixed RNG seed and makes the
 * sandbox AI's use of the pie rule understandable to humans watching the
 * opening.
 */
function shouldSandboxAIPickSwapSides(gameState: GameState): boolean {
  if (gameState.boardType !== 'square8') {
    return false;
  }

  const firstP1Placement = gameState.moveHistory.find(
    (m) => m.player === 1 && m.type === 'place_ring'
  );

  if (!firstP1Placement || !firstP1Placement.to) {
    return false;
  }

  const { x, y } = firstP1Placement.to;
  const size = gameState.board.size;
  const midLow = Math.floor((size - 1) / 2); // 3 on square8
  const midHigh = Math.ceil((size - 1) / 2); // 4 on square8

  const isCentral = (x === midLow || x === midHigh) && (y === midLow || y === midHigh);

  return isCentral;
}

function getLineDecisionMovesForSandboxAI(gameState: GameState, hooks: SandboxAIHooks): Move[] {
  if (gameState.currentPhase !== 'line_processing') {
    return [];
  }

  const allMoves = hooks.getValidMovesForCurrentPlayer();
  if (!Array.isArray(allMoves) || allMoves.length === 0) {
    return [];
  }

  // When explicit elimination moves are present (line‑reward debt),
  // mirror backend behaviour by preferring those over additional
  // process_line / choose_line_option options.
  const eliminationMoves = allMoves.filter((m) => m.type === 'eliminate_rings_from_stack');
  if (eliminationMoves.length > 0) {
    return eliminationMoves;
  }

  // RR-FIX-2025-12-13: Include no_line_action in decision moves. This move is
  // synthesized by SandboxOrchestratorAdapter.getValidMoves when there are no
  // lines to process, and the AI must apply it to advance to territory_processing.
  return allMoves.filter((m) => {
    const canonicalType = normalizeLegacyMoveType(m.type);
    return (
      canonicalType === 'process_line' ||
      canonicalType === 'choose_line_option' ||
      canonicalType === 'no_line_action' ||
      m.type === 'line_formation'
    );
  });
}

function getTerritoryDecisionMovesForSandboxAI(
  gameState: GameState,
  hooks: SandboxAIHooks
): Move[] {
  if (gameState.currentPhase !== 'territory_processing') {
    return [];
  }

  const allMoves = hooks.getValidMovesForCurrentPlayer();
  if (!Array.isArray(allMoves) || allMoves.length === 0) {
    return [];
  }

  // 1. Prefer explicit region‑processing moves when present, but treat
  //    skip_territory_processing as a sibling option so the AI can choose
  //    to defer additional region processing when the rules allow.
  const regionMoves = allMoves.filter((m) => {
    const canonicalType = normalizeLegacyMoveType(m.type);
    return canonicalType === 'choose_territory_option' || m.type === 'territory_claim';
  });
  const skipMoves = allMoves.filter((m) => m.type === 'skip_territory_processing');
  if (regionMoves.length > 0 || skipMoves.length > 0) {
    return [...regionMoves, ...skipMoves];
  }

  // 2. Otherwise, fall back to any explicit elimination decisions.
  const eliminationMoves = allMoves.filter((m) => m.type === 'eliminate_rings_from_stack');
  if (eliminationMoves.length > 0) {
    return eliminationMoves;
  }

  // RR-FIX-2025-12-13: Include no_territory_action in decision moves. This move is
  // synthesized by SandboxOrchestratorAdapter.getValidMoves when there are no
  // territory regions to process, and the AI must apply it to end the turn.
  return allMoves.filter((m) => m.type === 'no_territory_action');
}

export function buildSandboxMovementCandidates(
  gameState: GameState,
  hooks: SandboxAIHooks,
  rng: LocalAIRng
): { candidates: Move[]; debug: { captureCount: number; simpleMoveCount: number } } {
  // rng is currently unused but kept for parity with future heuristic policies.
  void rng;

  const playerNumber = gameState.currentPlayer;

  // Enumerate all available overtaking capture segments for this player.
  const captureSegments: Array<{
    from: Position;
    target: Position;
    landing: Position;
  }> = [];

  let stacks = hooks.getPlayerStacks(playerNumber, gameState.board);

  // If a placement occurred this turn, we must move the placed stack.
  const mustMoveFromStackKey = hooks.getMustMoveFromStackKey();
  if (mustMoveFromStackKey) {
    stacks = stacks.filter((s) => positionToString(s.position) === mustMoveFromStackKey);
  }

  for (const stack of stacks) {
    const segmentsFromStack = hooks.enumerateCaptureSegmentsFrom(stack.position, playerNumber);
    for (const seg of segmentsFromStack) {
      if (SANDBOX_AI_CAPTURE_DEBUG_ENABLED) {
        // eslint-disable-next-line no-console
        console.log(
          `Sandbox AI found capture: ${positionToString(seg.from)} -> ${positionToString(
            seg.landing
          )}`
        );
      }
      captureSegments.push(seg);
    }
  }

  // Enumerate simple non-capturing movement candidates.
  let landingCandidates = hooks.enumerateSimpleMovementLandings(playerNumber);

  // Enforce must-move semantics for simple movement as well: when a placement has
  // occurred this turn, only moves originating from the placed/updated stack are
  // eligible.
  if (mustMoveFromStackKey) {
    landingCandidates = landingCandidates.filter((m) => m.fromKey === mustMoveFromStackKey);
  }

  const debugCaptureCount = captureSegments.length;
  const debugSimpleMoveCount = landingCandidates.length;

  // Build canonical Move[] candidates for captures and simple movements. Within
  // each category we preserve deterministic tie-breaking (lexicographically
  // smallest landing position) to keep traces reproducible.
  const movementCandidates: Move[] = [];
  const baseMoveNumber = gameState.history.length + 1;

  // Helper to parse fromKey back into a Position.
  const stringToPositionLocal = (posStr: string): Position => {
    const parts = posStr.split(',').map(Number);
    if (parts.length === 2) {
      const [x, y] = parts;
      return { x, y };
    }
    if (parts.length === 3) {
      const [x, y, z] = parts;
      return { x, y, z };
    }
    return { x: 0, y: 0 };
  };

  captureSegments.forEach((seg, idx) => {
    movementCandidates.push({
      id: '',
      type: 'overtaking_capture',
      player: playerNumber,
      from: seg.from,
      captureTarget: seg.target,
      to: seg.landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: baseMoveNumber + idx,
    } as Move);
  });

  landingCandidates.forEach((cand, idx) => {
    movementCandidates.push({
      id: '',
      type: 'move_stack',
      player: playerNumber,
      from: stringToPositionLocal(cand.fromKey),
      to: cand.to,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: baseMoveNumber + captureSegments.length + idx,
    } as Move);
  });

  return {
    candidates: movementCandidates,
    debug: {
      captureCount: debugCaptureCount,
      simpleMoveCount: debugSimpleMoveCount,
    },
  };
}

/**
 * Select a move based on AI difficulty level.
 *
 * Difficulty affects move selection as follows:
 * - D1 (Beginner): Pure random selection
 * - D2-D3 (Learner/Casual): 70% random, 30% heuristic
 * - D4-D5 (Intermediate): 40% random, 60% heuristic
 * - D6-D7 (Advanced/Expert): 20% random, 80% heuristic
 * - D8-D10 (Strong Expert+): Pure heuristic selection
 *
 * This provides a smooth difficulty curve where lower levels make more
 * mistakes (random moves) while higher levels play more optimally.
 */
export function selectMoveWithDifficulty(
  playerNumber: number,
  gameState: GameState,
  candidates: Move[],
  rng: LocalAIRng,
  difficulty: number | undefined
): Move | null {
  if (candidates.length === 0) {
    return null;
  }

  if (candidates.length === 1) {
    return candidates[0];
  }

  // Default to D4 (intermediate) if no difficulty specified
  const effectiveDifficulty = difficulty ?? 4;

  // Calculate random move probability based on difficulty
  // D1: 100% random, D10: 0% random
  let randomProbability: number;
  if (effectiveDifficulty <= 1) {
    randomProbability = 1.0; // D1: Pure random
  } else if (effectiveDifficulty <= 3) {
    randomProbability = 0.7; // D2-D3: Mostly random
  } else if (effectiveDifficulty <= 5) {
    randomProbability = 0.4; // D4-D5: Mixed
  } else if (effectiveDifficulty <= 7) {
    randomProbability = 0.2; // D6-D7: Mostly heuristic
  } else {
    randomProbability = 0.0; // D8+: Pure heuristic
  }

  // Decide whether to use random or heuristic selection
  const useRandom = rng() < randomProbability;

  if (useRandom) {
    // Random selection: pick any candidate with equal probability
    const randomIndex = Math.floor(rng() * candidates.length);
    return candidates[randomIndex];
  }

  // Heuristic selection: use the standard move selector
  return chooseLocalMoveFromCandidates(playerNumber, gameState, candidates, rng);
}

export function selectSandboxMovementMove(
  gameState: GameState,
  candidates: Move[],
  rng: LocalAIRng,
  parityMode: boolean,
  difficulty?: number
): Move | null {
  const playerNumber = gameState.currentPlayer;

  if (parityMode) {
    // Parity mode: delegate directly to the shared selector so sandbox
    // movement/capture policy matches the backend local fallback.
    return chooseLocalMoveFromCandidates(playerNumber, gameState, candidates, rng);
  }

  // Use difficulty-aware selection when available
  return selectMoveWithDifficulty(playerNumber, gameState, candidates, rng, difficulty);
}

/**
 * Run a single AI turn in sandbox mode.
 *
 * All stochastic choices are driven by the injected `rng` parameter,
 * which must be a seeded RNG derived from GameState.rngSeed (typically
 * provided by ClientSandboxEngine). This keeps sandbox AI behaviour
 * deterministic for a fixed seed and aligned with backend fallback AI.
 *
 * Behaviour:
 * - In ring_placement: probabilistically decides between placing and
 *   skipping based on the ratio of placement vs non-placement
 *   options, then uses tryPlaceRings to apply the chosen placement.
 * - In movement:
 *   - Builds canonical overtaking_capture + move_stack candidates.
 *   - Uses chooseLocalMoveFromCandidates to choose between captures
 *     and simple moves in proportion to their counts.
 *   - Applies the chosen move via applyCanonicalMove, and auto-
 *     resolves any mandatory capture continuations while in the
 *     chain_capture phase.
 */
export async function maybeRunAITurnSandbox(hooks: SandboxAIHooks, rng: LocalAIRng): Promise<void> {
  // If we've exceeded the maximum consecutive no-op threshold, stop
  // executing AI turns to prevent infinite stalls and log spam.
  if (sandboxConsecutiveNoopAITurns >= SANDBOX_NOOP_MAX_THRESHOLD) {
    if (!sandboxStallLoggingSuppressed && SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED) {
      console.error(
        `[Sandbox AI] Stopping AI execution after ${sandboxConsecutiveNoopAITurns} consecutive no-op turns. Game is stalled.`
      );
      sandboxStallLoggingSuppressed = true;
    }
    return;
  }

  // Capture a pre-turn snapshot for history/event-sourcing. We rely on
  // the shared hashGameState helper so that backend and sandbox traces
  // are directly comparable.
  const beforeStateForHistory = hooks.getGameState();
  const beforeHashForHistory = hashGameState(beforeStateForHistory);

  // Reset last-move tracker at the start of each AI turn.
  hooks.setLastAIMove(null);

  let lastAIMove: Move | null = null;

  // Per-turn debug/diagnostic fields used for stall detection and trace logging.
  let debugIsAiTurn = false;
  let debugPlayerNumber: number | null = null;
  let debugBoardType: GameState['boardType'] | null = null;
  let debugPhaseBefore: GameState['currentPhase'] | null = null;
  let debugPlacementCandidateCount: number | null = null;
  let debugCaptureCount: number | null = null;
  let debugSimpleMoveCount: number | null = null;
  let debugForcedEliminationAttempted = false;
  let debugForcedEliminationEliminated = false;
  let debugAiDecisionSource: SandboxAiDecisionSource = 'local';
  let debugAiDifficultyRequested: number | null = null;
  let debugServiceAiType: string | null = null;
  let debugServiceDifficulty: number | null = null;
  let debugServiceHeuristicProfileId: string | null = null;
  let debugServiceUseNeuralNet: boolean | null = null;
  let debugServiceNnModelId: string | null = null;
  let debugServiceNnCheckpoint: string | null = null;
  let debugServiceNnueCheckpoint: string | null = null;
  let debugServiceThinkingTimeMs: number | null = null;
  let debugServiceError: string | null = null;

  try {
    const gameState = hooks.getGameState();
    const current = gameState.players.find((p) => p.playerNumber === gameState.currentPlayer);

    debugBoardType = gameState.boardType;
    debugPhaseBefore = gameState.currentPhase;
    debugPlayerNumber = gameState.currentPlayer;
    debugIsAiTurn = !!current && current.type === 'ai' && gameState.gameStatus === 'active';

    if (!current || current.type !== 'ai' || gameState.gameStatus !== 'active') {
      return;
    }

    // RR-FIX-2026-01-12: Log warning for very complex board states.
    // This helps users understand why the game might be slow on large boards
    // with many pieces. The move enumeration complexity is O(stacks × directions × cells).
    const stackCount = gameState.board.stacks.size;
    const isLargeBoard = gameState.boardType === 'square19' || gameState.boardType === 'hexagonal';
    if (stackCount > 80 || (isLargeBoard && stackCount > 50)) {
      // Only log once per 10 turns to avoid spamming the console
      if (gameState.history.length % 10 === 0) {
        console.debug(
          `[Sandbox AI] Complex board state: ${stackCount} stacks on ${gameState.boardType}. ` +
            'AI turns may take longer to compute.'
        );
      }
    }

    // RR-FIX-2026-01-12: Yield to browser before heavy computation.
    // This gives the browser a chance to process any pending events before
    // we start the expensive move enumeration.
    await new Promise((resolve) => window.setTimeout(resolve, 0));

    const parityMode = isSandboxAiParityModeEnabled();

    // Get AI difficulty for current player (if available)
    // Jan 10, 2026: Default to max difficulty (10) for optimal play
    const aiDifficulty = hooks.getAIDifficulty?.(current.playerNumber);
    const effectiveDifficulty = clampDifficulty(aiDifficulty, 10);
    debugAiDifficultyRequested = effectiveDifficulty;

    // Service-backed sandbox AI: when available, request a canonical move from
    // the backend which proxies to the Python AI service. This enables the
    // full difficulty ladder (minimax/mcts/descent + neural variants) for the
    // /sandbox host without embedding search engines in the client bundle.
    //
    // Parity mode intentionally bypasses the service so deterministic local
    // heuristics can be compared against backend fallback policies.
    if (!parityMode) {
      const stateForService = hooks.getGameState();
      const serviceResult = await tryRequestSandboxAIMove({
        state: serializeGameState(stateForService),
        difficulty: effectiveDifficulty,
        playerNumber: current.playerNumber,
      });

      if (serviceResult) {
        const serviceMove = serviceResult.move;
        debugServiceAiType = serviceResult.aiType ?? null;
        debugServiceDifficulty = serviceResult.difficulty ?? null;
        debugServiceHeuristicProfileId = serviceResult.heuristicProfileId ?? null;
        debugServiceUseNeuralNet = serviceResult.useNeuralNet ?? null;
        debugServiceNnModelId = serviceResult.nnModelId ?? null;
        debugServiceNnCheckpoint = serviceResult.nnCheckpoint ?? null;
        debugServiceNnueCheckpoint = serviceResult.nnueCheckpoint ?? null;
        debugServiceThinkingTimeMs = serviceResult.thinkingTimeMs ?? null;

        const candidates = hooks.getValidMovesForCurrentPlayer();
        const desiredKey = moveMatchKey(serviceMove);
        const matched = candidates.find((cand) => moveMatchKey(cand) === desiredKey);

        if (matched) {
          debugAiDecisionSource = 'service';
          const stateForMove = hooks.getGameState();
          const moveNumber = stateForMove.history.length + 1;
          const applied: Move = {
            ...matched,
            id: '',
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move;

          await hooks.applyCanonicalMove(applied);

          lastAIMove = applied;
          hooks.setLastAIMove(lastAIMove);
          return;
        }

        debugAiDecisionSource = 'mismatch';
        debugServiceError = 'service_move_not_in_candidates';
      } else {
        debugAiDecisionSource = 'unavailable';
        debugServiceError = 'service_unavailable';
      }
    }

    // === Pie rule (swap_sides) meta-move for 2-player sandbox AI ===
    // When the gate conditions match the backend pie rule and the opening
    // looks "strong" for Player 1, allow the sandbox AI playing as P2 to
    // invoke swap_sides once instead of continuing as normal.
    if (
      !parityMode &&
      gameState.currentPlayer === 2 &&
      hooks.canCurrentPlayerSwapSides() &&
      shouldSandboxAIPickSwapSides(gameState)
    ) {
      const applied = hooks.applySwapSidesForCurrentPlayer();
      if (applied) {
        const afterSwap = hooks.getGameState();
        const last = afterSwap.moveHistory[afterSwap.moveHistory.length - 1] ?? null;
        if (last) {
          lastAIMove = last;
          hooks.setLastAIMove(lastAIMove);
        }
        return;
      }
    }

    // === Ring placement phase: canonical candidates + shared selector ===
    if (gameState.currentPhase === 'ring_placement') {
      // RR-FIX-2026-01-11: Handle inconsistent state where mustMoveFromStackKey is set
      // during ring_placement phase. This can happen when a fixture is loaded mid-turn
      // or when state becomes corrupted. If mustMoveFromStackKey is set, placement has
      // already occurred and we just need to advance to movement phase.
      const mustMoveKey = hooks.getMustMoveFromStackKey();
      if (mustMoveKey) {
        // Placement already happened this turn - advance to movement via no_placement_action
        const stateForAdvance = hooks.getGameState();
        const moveNumber = stateForAdvance.history.length + 1;
        const advanceMove: Move = {
          type: 'no_placement_action',
          player: current.playerNumber,
          id: `no-placement-action-advance-${moveNumber}`,
          moveNumber,
          timestamp: new Date(),
          thinkTime: 0,
        } as Move;

        await hooks.applyCanonicalMove(advanceMove);
        lastAIMove = advanceMove;
        hooks.setLastAIMove(lastAIMove);
        return;
      }

      const ringsInHand = current.ringsInHand ?? 0;
      if (ringsInHand <= 0) {
        // With no rings in hand, ring_placement is not a real decision
        // phase under the rules: the player must either move from
        // existing stacks or, if completely blocked, undergo forced
        // elimination. Skip-placement moves with ringsInHand == 0 are
        // never legal; backend hosts advance phases without requiring
        // an explicit skip_placement decision.
        const beforeElimState = hooks.getGameState();
        const beforeElimHash = hashGameState(beforeElimState);

        const eliminated = hooks.maybeProcessForcedEliminationForCurrentPlayer();
        debugForcedEliminationAttempted = true;
        debugForcedEliminationEliminated = eliminated;

        const afterElimState = hooks.getGameState();
        const afterElimHash = hashGameState(afterElimState);

        if (
          !eliminated &&
          beforeElimHash === afterElimHash &&
          afterElimState.gameStatus === 'active'
        ) {
          // Forced elimination did not apply (player has legal moves from stacks).
          // The player is in ring_placement with ringsInHand=0 but has stacks.
          //
          // IMPORTANT: The orchestrator's getValidMoves() returns [] when
          // ringsInHand=0 in ring_placement phase (per RR-CANON-R076). Hosts
          // must construct an explicit no_placement_action move and apply it
          // to advance to movement phase.
          //
          // Check if current player has any movement or capture moves available
          // by directly enumerating them (bypassing the phase-locked getValidMoves).
          const playerStacks = hooks.getPlayerStacks(current.playerNumber, afterElimState.board);
          if (playerStacks.length > 0) {
            // First, construct and apply a no_placement_action move to advance
            // to movement phase. This is the canonical way to handle players
            // with 0 rings in ring_placement who have stacks on the board.
            const moveNumber = afterElimState.history.length + 1;
            const noPlacementMove: Move = {
              type: 'no_placement_action',
              player: current.playerNumber,
              id: `no-placement-action-${moveNumber}`,
              moveNumber,
              timestamp: new Date(),
              thinkTime: 0,
            } as Move;

            await hooks.applyCanonicalMove(noPlacementMove);
            lastAIMove = noPlacementMove;
            hooks.setLastAIMove(lastAIMove);
            return;
          }
          // No stacks on board either - fall through to let rest of function handle it
          return;
        } else {
          // Elimination was applied or game ended - return and let next iteration handle it
          return;
        }
      }

      // Mirror backend RuleEngine.validateSkipPlacement: skip is only
      // legal when placement is optional, i.e. the player both has
      // rings in hand and has at least one controlled stack with a
      // legal move/capture available.
      const boardForSkip = gameState.board;
      const playerStacksForSkip = hooks.getPlayerStacks(current.playerNumber, boardForSkip);
      const skipEligibility = evaluateSkipPlacementEligibilityAggregate(
        gameState,
        current.playerNumber
      );
      // Handle both aggregate (eligible) and legacy (canSkip) return shapes
      const skipResult = skipEligibility as { eligible?: boolean; canSkip?: boolean };
      const canSkipAggregate = skipResult.eligible ?? skipResult.canSkip ?? false;

      const placementCandidates = hooks.enumerateLegalRingPlacements(current.playerNumber);
      debugPlacementCandidateCount = placementCandidates.length;

      if (
        SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED &&
        !sandboxStallLoggingSuppressed &&
        placementCandidates.length > 0
      ) {
        // eslint-disable-next-line no-console
        console.log(
          '[Sandbox AI Debug] Placement candidates before filtering:',
          JSON.stringify({
            count: placementCandidates.length,
            player: current.playerNumber,
            ringsInHand,
            skipEligible: canSkipAggregate,
            playerStacksCount: playerStacksForSkip.length,
          })
        );
      }

      // If no legal placement under sandbox no-dead-placement, we may still
      // skip placement when backend would also allow it; otherwise, the
      // state is structurally blocked.
      if (placementCandidates.length === 0) {
        if (canSkipAggregate) {
          const stateForMove = hooks.getGameState();
          const moveNumber = stateForMove.history.length + 1;
          // For skip_placement, use the canonical sentinel {0,0}. The
          // coordinate carries no spatial meaning but must be stable across
          // engines for trace/parity tooling.
          const sentinelTo: Position = { x: 0, y: 0 };

          const skipMove: Move = {
            id: '',
            type: 'skip_placement',
            player: current.playerNumber,
            to: sentinelTo,
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move;

          await hooks.applyCanonicalMove(skipMove);

          lastAIMove = skipMove;
          hooks.setLastAIMove(lastAIMove);

          return;
        }

        // Otherwise placement is mandatory or state is structurally blocked under the
        // sandbox no-dead-placement rule. Delegate to the same forced-elimination
        // helper used in the movement phase so that structurally stuck positions
        // still progress via cap elimination + turn advancement instead of
        // leaving the AI turn as a no-op.
        const beforeElimState = hooks.getGameState();
        const beforeElimHash = hashGameState(beforeElimState);

        const eliminated = hooks.maybeProcessForcedEliminationForCurrentPlayer();
        debugForcedEliminationAttempted = true;
        debugForcedEliminationEliminated = eliminated;

        const afterElimState = hooks.getGameState();
        const afterElimHash = hashGameState(afterElimState);

        if (
          SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED &&
          !sandboxStallLoggingSuppressed &&
          !eliminated &&
          beforeElimHash === afterElimHash &&
          afterElimState.gameStatus === 'active'
        ) {
          console.warn(
            '[Sandbox AI Stall Diagnostic] Ring-placement forced elimination did not change state',
            {
              boardType: gameState.boardType,
              currentPlayer: gameState.currentPlayer,
              currentPhase: gameState.currentPhase,
              ringsInHand: gameState.players.map((p) => ({
                playerNumber: p.playerNumber,
                type: p.type,
                ringsInHand: p.ringsInHand,
                stacks: hooks.getPlayerStacks(p.playerNumber, gameState.board).length,
              })),
            }
          );
        }

        return;
      }

      const board = gameState.board;
      const boardConfig = BOARD_CONFIGS[gameState.boardType];
      const stacksForPlayer = hooks.getPlayerStacks(current.playerNumber, board);
      const ringsOnBoard = stacksForPlayer.reduce((sum, stack) => sum + stack.rings.length, 0);

      const perPlayerCap = boardConfig.ringsPerPlayer;
      const remainingByCap = perPlayerCap - ringsOnBoard;
      const remainingBySupply = ringsInHand;
      const maxAvailableGlobal = Math.min(remainingByCap, remainingBySupply);

      // Determine whether the player has any legal moves or captures from
      // existing stacks if they choose to skip placement. This mirrors the
      // logic used in the placementCandidates === 0 branch above.
      const hasAnyActionFromStacks = (() => {
        for (const stack of stacksForPlayer) {
          if (hooks.hasAnyLegalMoveOrCaptureFrom(stack.position, current.playerNumber, board)) {
            return true;
          }
        }
        return false;
      })();

      if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
        // eslint-disable-next-line no-console
        console.log(
          '[Sandbox AI Debug] Ring supply calculation:',
          JSON.stringify({
            player: current.playerNumber,
            ringsOnBoard,
            perPlayerCap,
            remainingByCap,
            remainingBySupply,
            maxAvailableGlobal,
          })
        );
      }

      // When the player hits their ring cap (maxAvailableGlobal = 0), they can
      // still skip placement if they have legal moves from existing stacks and
      // the aggregate reports that skipping is legal.
      if (maxAvailableGlobal <= 0) {
        if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
          console.warn(
            '[Sandbox AI Debug] maxAvailableGlobal <= 0, checking for skip_placement option'
          );
        }

        if (canSkipAggregate) {
          // Player has hit cap but can skip placement and move existing stacks.
          // As with other skip_placement moves, use the canonical sentinel {0,0}
          // so backend and sandbox traces share the same representation.
          const stateForMove = hooks.getGameState();
          const moveNumber = stateForMove.history.length + 1;
          const sentinelTo: Position = { x: 0, y: 0 };

          const skipMove: Move = {
            id: '',
            type: 'skip_placement',
            player: current.playerNumber,
            to: sentinelTo,
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move;

          await hooks.applyCanonicalMove(skipMove);

          lastAIMove = skipMove;
          hooks.setLastAIMove(lastAIMove);

          return;
        }

        // Otherwise, no moves available - attempt forced elimination
        const beforeElimState = hooks.getGameState();
        const beforeElimHash = hashGameState(beforeElimState);

        const eliminated = hooks.maybeProcessForcedEliminationForCurrentPlayer();
        debugForcedEliminationAttempted = true;
        debugForcedEliminationEliminated = eliminated;

        const afterElimState = hooks.getGameState();
        const afterElimHash = hashGameState(afterElimState);

        if (
          SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED &&
          !sandboxStallLoggingSuppressed &&
          !eliminated &&
          beforeElimHash === afterElimHash &&
          afterElimState.gameStatus === 'active'
        ) {
          console.warn(
            '[Sandbox AI Stall Diagnostic] At ring cap, forced elimination did not change state',
            {
              boardType: gameState.boardType,
              currentPlayer: gameState.currentPlayer,
              currentPhase: gameState.currentPhase,
              ringsInHand: gameState.players.map((p) => ({
                playerNumber: p.playerNumber,
                type: p.type,
                ringsInHand: p.ringsInHand,
                stacks: hooks.getPlayerStacks(p.playerNumber, gameState.board).length,
              })),
            }
          );
        }

        return;
      }

      const candidates: Move[] = [];
      const baseMoveNumber = gameState.history.length + 1;

      // Use the same no-dead-placement semantics as the backend
      // RuleEngine.validateRingPlacement by filtering multi-ring
      // candidates through a hypothetical-board + reachability
      // check. This keeps sandbox AI placement counts aligned with
      // backend getValidMoves so parity/trace harnesses can replay
      // sandbox traces without introducing stack-height drift.
      let filteredOutCount = 0;
      for (const pos of placementCandidates) {
        const key = positionToString(pos);
        const existing = board.stacks.get(key);
        const isOccupied = !!existing && existing.rings.length > 0;
        const maxPerPlacement = isOccupied ? 1 : Math.min(3, maxAvailableGlobal);

        for (let count = 1; count <= maxPerPlacement; count++) {
          const hypotheticalBoard = hooks.createHypotheticalBoardWithPlacement(
            board,
            pos,
            current.playerNumber,
            count
          );

          const hasActionFromStack = hooks.hasAnyLegalMoveOrCaptureFrom(
            pos,
            current.playerNumber,
            hypotheticalBoard
          );

          if (!hasActionFromStack) {
            filteredOutCount++;
            continue;
          }

          candidates.push({
            id: '',
            type: 'place_ring',
            player: current.playerNumber,
            to: pos,
            placementCount: count,
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: baseMoveNumber + candidates.length,
          } as Move);
        }
      }

      if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
        // eslint-disable-next-line no-console
        console.log(
          '[Sandbox AI Debug] After multi-ring filtering:',
          JSON.stringify({
            initialCandidates: placementCandidates.length,
            finalCandidates: candidates.length,
            filteredOut: filteredOutCount,
            hasSkipOption: hasAnyActionFromStacks,
            ringsInHand,
            maxAvailableGlobal,
          })
        );
      }

      if (hasAnyActionFromStacks && canSkipAggregate) {
        // For canonical skip_placement candidates, always use the shared
        // sentinel {0,0}. The destination coordinate is ignored by rules
        // logic but must be consistent across engines for trace parity.
        const sentinelTo: Position = { x: 0, y: 0 };

        candidates.push({
          id: '',
          type: 'skip_placement',
          player: current.playerNumber,
          from: undefined,
          to: sentinelTo,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: baseMoveNumber + candidates.length,
        } as Move);
      }

      if (candidates.length === 0) {
        // At this point, rings are in hand and placement is still
        // mandatory (hasAnyActionFromStacks === false), but every
        // raw placement candidate has been filtered out by the
        // no-dead-placement check. This mirrors a true "blocked with
        // stacks" situation under the real rules and must be resolved
        // via forced elimination rather than leaving the AI turn as a
        // structural no-op that will be detected as a stall.
        const beforeElimState = hooks.getGameState();
        const beforeElimHash = hashGameState(beforeElimState);

        const eliminated = hooks.maybeProcessForcedEliminationForCurrentPlayer();
        debugForcedEliminationAttempted = true;
        debugForcedEliminationEliminated = eliminated;

        const afterElimState = hooks.getGameState();
        const afterElimHash = hashGameState(afterElimState);

        if (
          SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED &&
          !sandboxStallLoggingSuppressed &&
          !eliminated &&
          beforeElimHash === afterElimHash &&
          afterElimState.gameStatus === 'active'
        ) {
          console.warn(
            '[Sandbox AI Stall Diagnostic] Ring-placement forced elimination (no safe placements) did not change state',
            {
              boardType: gameState.boardType,
              currentPlayer: gameState.currentPlayer,
              currentPhase: gameState.currentPhase,
              ringsInHand: gameState.players.map((p) => ({
                playerNumber: p.playerNumber,
                type: p.type,
                ringsInHand: p.ringsInHand,
                stacks: hooks.getPlayerStacks(p.playerNumber, gameState.board).length,
              })),
            }
          );
        }

        return;
      }

      // Use difficulty-aware selection in non-parity mode
      const selected = parityMode
        ? chooseLocalMoveFromCandidates(current.playerNumber, gameState, candidates, rng)
        : selectMoveWithDifficulty(current.playerNumber, gameState, candidates, rng, aiDifficulty);

      if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
        // eslint-disable-next-line no-console
        console.log(
          '[Sandbox AI Debug] Move selection result:',
          JSON.stringify({
            selected: selected
              ? {
                  type: selected.type,
                  to: selected.to
                    ? `${selected.to.x},${selected.to.y}${selected.to.z !== undefined ? ',' + selected.to.z : ''}`
                    : null,
                  count: selected.placementCount,
                }
              : null,
            candidatesLength: candidates.length,
            candidateTypes: candidates.map((c) => c.type),
            skipCount: candidates.filter((c) => c.type === 'skip_placement').length,
            placeCount: candidates.filter((c) => c.type === 'place_ring').length,
            aiDifficulty,
          })
        );
      }

      if (!selected) {
        console.error(
          '[Sandbox AI] selectMoveWithDifficulty returned null with',
          candidates.length,
          'candidates'
        );
        return;
      }

      // RR-FIX-2026-01-12: Handle placement validation failures gracefully.
      // When a place_ring move fails validation (e.g., no-dead-placement rule),
      // remove the failed candidate and retry with another move. This prevents
      // AI vs AI games from getting stuck when the sandbox's pre-check differs
      // slightly from the orchestrator's validation.
      let remainingCandidates = [...candidates];
      let currentSelection: Move | null = selected;
      let attempts = 0;
      const maxAttempts = remainingCandidates.length + 1;

      while (attempts < maxAttempts && remainingCandidates.length > 0 && currentSelection) {
        attempts++;

        const stateForMove = hooks.getGameState();
        const moveNumber = stateForMove.history.length + 1;

        let moveToApply: Move | null = null;

        if (currentSelection.type === 'skip_placement') {
          moveToApply = {
            id: '',
            type: 'skip_placement',
            player: current.playerNumber,
            from: undefined,
            // Preserve the sentinel position chosen when we fabricated the
            // skip_placement candidate so traces match the Python engine.
            to: currentSelection.to ?? ({ x: 0, y: 0 } as Position),
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move;
        } else if (currentSelection.type === 'place_ring') {
          // Removed the `&& selected.to` check that was causing stalls when selected.to
          // was somehow missing/corrupted. Instead, provide a defensive fallback.
          if (!currentSelection.to) {
            console.error('[Sandbox AI] place_ring selected but to is missing:', currentSelection);
            // Remove this invalid candidate and try another
            remainingCandidates = remainingCandidates.filter((c) => c !== currentSelection);
            if (remainingCandidates.length > 0) {
              currentSelection = parityMode
                ? chooseLocalMoveFromCandidates(
                    current.playerNumber,
                    gameState,
                    remainingCandidates,
                    rng
                  )
                : selectMoveWithDifficulty(
                    current.playerNumber,
                    gameState,
                    remainingCandidates,
                    rng,
                    aiDifficulty
                  );
              if (!currentSelection) break;
              continue;
            }
            break;
          }
          moveToApply = {
            id: '',
            type: 'place_ring',
            player: current.playerNumber,
            from: undefined,
            to: currentSelection.to,
            placementCount: currentSelection.placementCount ?? 1,
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move;
        } else {
          // Unexpected move type in ring_placement; log for debugging.
          console.error(
            '[Sandbox AI] Unexpected move type in ring_placement:',
            currentSelection.type
          );
          return;
        }

        try {
          await hooks.applyCanonicalMove(moveToApply);
          lastAIMove = moveToApply;
          hooks.setLastAIMove(lastAIMove);
          return;
        } catch (error) {
          // RR-FIX-2026-01-12: Placement validation failed. This can happen when:
          // 1. The no-dead-placement check in the orchestrator differs slightly from
          //    the sandbox's pre-check (e.g., race conditions, subtle view differences)
          // 2. The board state changed between candidate enumeration and move application
          //
          // Remove this candidate and try another move from the remaining candidates.
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.warn(
            `[Sandbox AI] Placement validation failed for ${moveToApply.type} at ` +
              `(${moveToApply.to?.x},${moveToApply.to?.y}): ${errorMessage}. ` +
              `Trying another candidate (${remainingCandidates.length - 1} remaining).`
          );

          // Remove the failed candidate
          // Capture moveToApply in a const for type safety inside the filter callback
          const failedMove = moveToApply;
          if (!failedMove) continue; // Should never happen, but satisfies type checker
          remainingCandidates = remainingCandidates.filter((c) => {
            if (c.type !== failedMove.type) return true;
            if (c.type === 'place_ring' && failedMove.type === 'place_ring') {
              // Remove candidates at the same position with same or higher count
              // since they're likely to fail too
              const samePos =
                c.to &&
                failedMove.to &&
                c.to.x === failedMove.to.x &&
                c.to.y === failedMove.to.y &&
                (c.to.z === undefined || c.to.z === failedMove.to.z);
              if (samePos && (c.placementCount ?? 1) >= (failedMove.placementCount ?? 1)) {
                return false;
              }
            }
            return true;
          });

          if (remainingCandidates.length === 0) {
            // All candidates exhausted - fall back to forced elimination
            console.warn(
              '[Sandbox AI] All placement candidates failed validation. Falling back to forced elimination.'
            );
            const eliminated = hooks.maybeProcessForcedEliminationForCurrentPlayer();
            if (!eliminated) {
              console.error('[Sandbox AI] Forced elimination also failed. Game may be stuck.');
            }
            return;
          }

          // Select another candidate from the remaining list
          currentSelection = parityMode
            ? chooseLocalMoveFromCandidates(
                current.playerNumber,
                gameState,
                remainingCandidates,
                rng
              )
            : selectMoveWithDifficulty(
                current.playerNumber,
                gameState,
                remainingCandidates,
                rng,
                aiDifficulty
              );

          if (!currentSelection) {
            console.warn(
              '[Sandbox AI] No valid candidate selected after filtering. Trying forced elimination.'
            );
            hooks.maybeProcessForcedEliminationForCurrentPlayer();
            return;
          }
        }
      }

      // If we get here without returning, all attempts failed
      console.warn('[Sandbox AI] Exhausted all placement attempts. Trying forced elimination.');
      hooks.maybeProcessForcedEliminationForCurrentPlayer();
      return;
    }

    // === Line and territory decision phases: canonical decision moves ===
    if (
      gameState.currentPhase === 'line_processing' ||
      gameState.currentPhase === 'territory_processing'
    ) {
      let decisionCandidates: Move[] = [];

      if (gameState.currentPhase === 'line_processing') {
        // When a line-reward elimination is pending for the current player,
        // mirror backend GameEngine.getValidMoves behaviour by surfacing
        // explicit eliminate_rings_from_stack decisions instead of further
        // process_line / choose_line_option moves. This keeps the sandbox AI
        // aligned with move-driven line-processing semantics and ensures that
        // seed-based traces express the same high-level action sequence as the
        // backend (collapse line -> eliminate cap -> continue lines/territory).
        if (hooks.hasPendingLineRewardElimination()) {
          const state = hooks.getGameState();
          const currentPlayer = state.currentPlayer;
          const board = state.board;
          const playerStacks = hooks.getPlayerStacks(currentPlayer, board);

          if (playerStacks.length > 0) {
            const baseMoveNumber = state.history.length + 1;
            playerStacks.forEach((stack, idx) => {
              const key = positionToString(stack.position);
              decisionCandidates.push({
                id: `eliminate-${key}`,
                type: 'eliminate_rings_from_stack',
                player: currentPlayer,
                to: stack.position,
                timestamp: new Date(),
                thinkTime: 0,
                moveNumber: baseMoveNumber + idx,
              } as Move);
            });
          }
        } else {
          decisionCandidates = getLineDecisionMovesForSandboxAI(gameState, hooks);
        }
      } else {
        decisionCandidates = getTerritoryDecisionMovesForSandboxAI(gameState, hooks);
      }

      if (decisionCandidates.length === 0) {
        // RR-FIX-2025-12-27: Handle cases where getValidMoves returns empty but
        // decision phases need to advance. This prevents AI hangs during ring_elimination
        // and region_order decision phases.
        if (gameState.currentPhase === 'territory_processing') {
          // Check if there's a pending territory self-elimination decision
          if (hooks.hasPendingTerritorySelfElimination()) {
            // Manually find a stack to eliminate
            const currentPlayer = gameState.currentPlayer;
            const playerStacks = hooks.getPlayerStacks(currentPlayer, gameState.board);
            const eligibleStack = playerStacks.find((s) => s.stackHeight > 0);

            if (eligibleStack) {
              const capHeight = eligibleStack.capHeight || 0;
              const count = Math.max(1, capHeight);
              const fallbackMove: Move = {
                id: `territory-elim-fallback-${Date.now()}`,
                type: 'eliminate_rings_from_stack',
                player: currentPlayer,
                to: eligibleStack.position,
                eliminatedRings: [{ player: currentPlayer, count }],
                eliminationFromStack: {
                  position: eligibleStack.position,
                  capHeight,
                  totalHeight: eligibleStack.stackHeight,
                },
                timestamp: new Date(),
                thinkTime: 0,
                moveNumber: gameState.history.length + 1,
              } as Move;

              if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
                console.debug(
                  '[Sandbox AI Debug] territory_processing fallback: applying eliminate_rings_from_stack',
                  {
                    position: eligibleStack.position,
                    capHeight,
                    stackHeight: eligibleStack.stackHeight,
                  }
                );
              }

              try {
                await hooks.applyCanonicalMove(fallbackMove);
                lastAIMove = fallbackMove;
                hooks.setLastAIMove(lastAIMove);
                return;
              } catch (e) {
                if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
                  console.error(
                    '[Sandbox AI Debug] territory_processing elimination fallback failed',
                    e
                  );
                }
              }
            }
          }

          // No pending elimination or fallback failed - synthesize no_territory_action
          const moveNumber = gameState.history.length + 1;
          const noTerritoryMove: Move = {
            id: `no-territory-action-ai-${moveNumber}`,
            type: 'no_territory_action',
            player: gameState.currentPlayer,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move;

          if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
            console.debug(
              '[Sandbox AI Debug] territory_processing fallback: applying no_territory_action'
            );
          }

          try {
            await hooks.applyCanonicalMove(noTerritoryMove);
            lastAIMove = noTerritoryMove;
            hooks.setLastAIMove(lastAIMove);
            return;
          } catch (e) {
            if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
              console.error('[Sandbox AI Debug] no_territory_action fallback failed', e);
            }
          }
        } else if (gameState.currentPhase === 'line_processing') {
          // Check if there's a pending line reward elimination
          if (hooks.hasPendingLineRewardElimination()) {
            // Manually find a stack to eliminate
            const currentPlayer = gameState.currentPlayer;
            const playerStacks = hooks.getPlayerStacks(currentPlayer, gameState.board);
            const eligibleStack = playerStacks.find((s) => s.stackHeight > 0);

            if (eligibleStack) {
              const capHeight = eligibleStack.capHeight || 0;
              const count = Math.max(1, capHeight);
              const fallbackMove: Move = {
                id: `line-elim-fallback-${Date.now()}`,
                type: 'eliminate_rings_from_stack',
                player: currentPlayer,
                to: eligibleStack.position,
                eliminatedRings: [{ player: currentPlayer, count }],
                eliminationFromStack: {
                  position: eligibleStack.position,
                  capHeight,
                  totalHeight: eligibleStack.stackHeight,
                },
                timestamp: new Date(),
                thinkTime: 0,
                moveNumber: gameState.history.length + 1,
              } as Move;

              if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
                console.debug(
                  '[Sandbox AI Debug] line_processing fallback: applying eliminate_rings_from_stack',
                  {
                    position: eligibleStack.position,
                    capHeight,
                    stackHeight: eligibleStack.stackHeight,
                  }
                );
              }

              try {
                await hooks.applyCanonicalMove(fallbackMove);
                lastAIMove = fallbackMove;
                hooks.setLastAIMove(lastAIMove);
                return;
              } catch (e) {
                if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
                  console.error(
                    '[Sandbox AI Debug] line_processing elimination fallback failed',
                    e
                  );
                }
              }
            }
          }

          // No pending elimination or fallback failed - synthesize no_line_action
          const moveNumber = gameState.history.length + 1;
          const noLineMove: Move = {
            id: `no-line-action-ai-${moveNumber}`,
            type: 'no_line_action',
            player: gameState.currentPlayer,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move;

          if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
            console.debug('[Sandbox AI Debug] line_processing fallback: applying no_line_action');
          }

          try {
            await hooks.applyCanonicalMove(noLineMove);
            lastAIMove = noLineMove;
            hooks.setLastAIMove(lastAIMove);
            return;
          } catch (e) {
            if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
              console.error('[Sandbox AI Debug] no_line_action fallback failed', e);
            }
          }
        }

        // If all fallbacks failed, return without progress
        return;
      }

      // Use difficulty-aware selection in non-parity mode
      const selectedDecision = parityMode
        ? chooseLocalMoveFromCandidates(gameState.currentPlayer, gameState, decisionCandidates, rng)
        : selectMoveWithDifficulty(
            gameState.currentPlayer,
            gameState,
            decisionCandidates,
            rng,
            aiDifficulty
          );

      if (!selectedDecision) {
        return;
      }

      const stateForMove = hooks.getGameState();
      const moveNumber = stateForMove.history.length + 1;

      const decisionMove: Move = {
        ...selectedDecision,
        id: '',
        moveNumber,
        timestamp: new Date(),
      } as Move;

      await hooks.applyCanonicalMove(decisionMove);

      lastAIMove = decisionMove;
      hooks.setLastAIMove(lastAIMove);
      return;
    }

    // === Forced elimination phase: canonical forced_elimination decision ===
    if (gameState.currentPhase === 'forced_elimination') {
      const allMoves = hooks.getValidMovesForCurrentPlayer();
      const forcedEliminationMoves = Array.isArray(allMoves)
        ? allMoves.filter((m) => m.type === 'forced_elimination')
        : [];

      if (forcedEliminationMoves.length === 0) {
        if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
          console.warn(
            '[Sandbox AI Debug] forced_elimination phase but no forced_elimination moves; attempting manual fallback',
            {
              boardType: gameState.boardType,
              currentPlayer: gameState.currentPlayer,
              currentPhase: gameState.currentPhase,
              allMovesCount: allMoves.length,
              allMovesTypes: allMoves.map((m) => m.type),
            }
          );
        }

        // Fallback: if the engine reports no FE moves (likely due to inconsistent state
        // where hasForcedEliminationAction is false because other moves exist, but
        // phase is forced_elimination), manually find a stack and eliminate it to
        // unstick the game.
        const stacks = hooks.getPlayerStacks(gameState.currentPlayer, gameState.board);
        const eligibleStack = stacks.find((s) => s.stackHeight > 0);

        if (eligibleStack) {
          const capHeight = eligibleStack.capHeight || 0;
          const count = Math.max(1, capHeight);
          const fallbackMove: Move = {
            id: `forced-elim-fallback-${Date.now()}`,
            // Canonical forced elimination move (handled by turnOrchestrator).
            type: 'forced_elimination',
            player: gameState.currentPlayer,
            to: eligibleStack.position,
            eliminatedRings: [{ player: gameState.currentPlayer, count }],
            eliminationFromStack: {
              position: eligibleStack.position,
              capHeight,
              totalHeight: eligibleStack.stackHeight,
            },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: gameState.history.length + 1,
          } as Move;

          try {
            await hooks.applyCanonicalMove(fallbackMove);
            hooks.setLastAIMove(fallbackMove);
            return;
          } catch (e) {
            console.error('[Sandbox AI Debug] Manual fallback FE move failed', e);
          }
        }

        return;
      }

      // Use difficulty-aware selection in non-parity mode
      let selectedForcedElimination = parityMode
        ? chooseLocalMoveFromCandidates(
            gameState.currentPlayer,
            gameState,
            forcedEliminationMoves,
            rng
          )
        : selectMoveWithDifficulty(
            gameState.currentPlayer,
            gameState,
            forcedEliminationMoves,
            rng,
            aiDifficulty
          );

      if (!selectedForcedElimination) {
        // Harden FE behaviour: when canonical candidates exist, never leave the
        // turn as a structural no-op just because the local selector declined
        // to pick one. Fall back to a simple deterministic/RNG-based choice so
        // that stall detection will always observe progress in forced_elimination
        // states where hasForcedEliminationAction === true.
        if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
          console.warn(
            '[Sandbox AI Debug] selectMoveWithDifficulty returned null for forced_elimination; falling back to direct selection',
            {
              candidateCount: forcedEliminationMoves.length,
              boardType: gameState.boardType,
              currentPlayer: gameState.currentPlayer,
              aiDifficulty,
            }
          );
        }

        const fallbackIndex =
          forcedEliminationMoves.length === 1
            ? 0
            : Math.floor(rng() * forcedEliminationMoves.length);
        selectedForcedElimination = forcedEliminationMoves[fallbackIndex];
      }

      const stateForMove = hooks.getGameState();
      const moveNumber = stateForMove.history.length + 1;

      const forcedEliminationMove: Move = {
        ...selectedForcedElimination,
        id: '',
        moveNumber,
        timestamp: new Date(),
        // Preserve canonical forced_elimination type; the shared orchestrator
        // accepts this MoveType directly.
        type: 'forced_elimination',
      } as Move;

      debugForcedEliminationAttempted = true;
      debugForcedEliminationEliminated = true;

      try {
        await hooks.applyCanonicalMove(forcedEliminationMove);
      } catch (e) {
        console.error('[Sandbox AI Debug] applyCanonicalMove failed for forced_elimination', e);
        throw e;
      }

      lastAIMove = forcedEliminationMove;
      hooks.setLastAIMove(lastAIMove);
      return;
    }

    // === Movement / capture phase: canonical capture + movement candidates ===
    // Re-read state to ensure we have the latest phase/player after any
    // prior phase processing (e.g., ring placement may have advanced phases).
    const movementState = hooks.getGameState();
    const movementPlayer = movementState.players.find(
      (p) => p.playerNumber === movementState.currentPlayer
    );

    // Early exit if phase has advanced past movement/capture or it's not an AI's turn
    if (
      (movementState.currentPhase !== 'movement' && movementState.currentPhase !== 'capture') ||
      !movementPlayer ||
      movementPlayer.type !== 'ai' ||
      movementState.gameStatus !== 'active'
    ) {
      return;
    }

    const playerNumber = movementPlayer.playerNumber;

    // Enumerate canonical movement/capture candidates via the host's
    // getValidMoves surface so sandbox AI stays aligned with backend
    // GameEngine.getValidMoves semantics.
    const allMoves = hooks.getValidMovesForCurrentPlayer();
    let movementCandidates: Move[] = [];

    if (movementState.currentPhase === 'movement') {
      movementCandidates = allMoves.filter((m) => {
        const canonicalType = normalizeLegacyMoveType(m.type);
        return canonicalType === 'overtaking_capture' || canonicalType === 'move_stack';
      });
      // Fallback: if the host-reported canonical surface has no movement/capture
      // moves but shared movement/capture helpers still see legal actions, use
      // the shared helpers to build candidates. This defends against rare
      // enumeration mismatches in deep seeds while keeping the primary path
      // aligned with backend getValidMoves().
      if (movementCandidates.length === 0) {
        const fallback = buildSandboxMovementCandidates(movementState, hooks, rng);
        if (fallback.candidates.length > 0) {
          movementCandidates = fallback.candidates;
        }
      }
    } else {
      // capture phase: consider overtaking_capture vs skip_capture.
      movementCandidates = allMoves.filter(
        (m) => m.type === 'overtaking_capture' || m.type === 'skip_capture'
      );
    }

    const captureCount = movementCandidates.filter((m) => m.type === 'overtaking_capture').length;
    const simpleMoveCount = movementCandidates.filter(
      (m) => normalizeLegacyMoveType(m.type) === 'move_stack'
    ).length;

    debugCaptureCount = captureCount;
    debugSimpleMoveCount = simpleMoveCount;

    if (movementCandidates.length === 0) {
      const mustMoveFromStackKey = hooks.getMustMoveFromStackKey();
      let stacksForDebug = hooks.getPlayerStacks(playerNumber, movementState.board);
      if (mustMoveFromStackKey) {
        stacksForDebug = stacksForDebug.filter(
          (s) => positionToString(s.position) === mustMoveFromStackKey
        );
      }

      if (SANDBOX_AI_CAPTURE_DEBUG_ENABLED) {
        const stackDetails = stacksForDebug.map((s) => ({
          pos: positionToString(s.position),
          height: s.stackHeight,
          cap: s.capHeight,
          rings: s.rings,
          hasAnyAction: hooks.hasAnyLegalMoveOrCaptureFrom(
            s.position,
            playerNumber,
            movementState.board
          ),
        }));
        // eslint-disable-next-line no-console
        console.log('[Sandbox AI Debug] No moves found', {
          phase: movementState.currentPhase,
          mustMoveFromStackKey,
          stacksCount: stacksForDebug.length,
          captureCount,
          landingCount: simpleMoveCount,
          player: playerNumber,
          stackDetails,
        });
      }

      // Only attempt forced elimination / structural resolution when we
      // are in movement phase. In capture phase, an empty canonical
      // move surface would indicate a deeper host bug; for now treat it
      // as a no-op so stall diagnostics can surface it.
      if (movementState.currentPhase === 'movement') {
        const beforeElimState = hooks.getGameState();
        const beforeElimHash = hashGameState(beforeElimState);

        const eliminated = hooks.maybeProcessForcedEliminationForCurrentPlayer();
        debugForcedEliminationAttempted = true;
        debugForcedEliminationEliminated = eliminated;

        const afterElimState = hooks.getGameState();
        const afterElimHash = hashGameState(afterElimState);

        if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
          if (
            !eliminated &&
            beforeElimHash === afterElimHash &&
            afterElimState.gameStatus === 'active'
          ) {
            console.warn(
              '[Sandbox AI Stall Diagnostic] No captures/moves, forced elimination did not change state',
              {
                boardType: movementState.boardType,
                currentPlayer: movementState.currentPlayer,
                currentPhase: movementState.currentPhase,
                ringsInHand: movementState.players.map((p) => ({
                  playerNumber: p.playerNumber,
                  type: p.type,
                  ringsInHand: p.ringsInHand,
                  stacks: hooks.getPlayerStacks(p.playerNumber, movementState.board).length,
                })),
              }
            );
          }
        }
      }

      return;
    }

    const selectedMove = selectSandboxMovementMove(
      movementState,
      movementCandidates,
      rng,
      parityMode,
      aiDifficulty
    );

    if (!selectedMove) {
      return;
    }

    if (selectedMove.type === 'overtaking_capture') {
      const stateBeforeCapture = hooks.getGameState();

      // Validate we're still in the expected phase and it's still our turn before applying
      if (
        stateBeforeCapture.currentPlayer !== selectedMove.player ||
        (stateBeforeCapture.currentPhase !== 'movement' &&
          stateBeforeCapture.currentPhase !== 'capture')
      ) {
        // State has changed since we enumerated moves - abort this iteration
        return;
      }

      const firstMoveNumber = stateBeforeCapture.history.length + 1;

      const firstMove: Move = {
        ...selectedMove,
        id: '',
        moveNumber: firstMoveNumber,
        timestamp: new Date(),
      } as Move;

      await hooks.applyCanonicalMove(firstMove);

      lastAIMove = firstMove;
      hooks.setLastAIMove(lastAIMove);

      let chainPosition: Position = firstMove.to as Position;
      const MAX_CHAIN_STEPS = 32;
      let steps = 0;

      while (true) {
        const stateAfter = hooks.getGameState();

        if (stateAfter.gameStatus !== 'active' || stateAfter.currentPhase !== 'chain_capture') {
          break;
        }

        steps++;
        if (steps > MAX_CHAIN_STEPS) {
          // Defensive: avoid infinite loops if something goes wrong with
          // chain-capture phase transitions.

          console.warn(
            '[Sandbox AI] Aborting chain_capture auto-resolution after MAX_CHAIN_STEPS',
            {
              boardType: stateAfter.boardType,
              currentPlayer: stateAfter.currentPlayer,
              currentPhase: stateAfter.currentPhase,
            }
          );
          break;
        }

        // RR-FIX-2026-01-12: Yield to browser during chain capture resolution.
        // On large boards with long capture chains, each iteration calls
        // enumerateCaptureSegmentsFrom which can be expensive. Yielding every
        // few iterations keeps the browser responsive.
        if (steps % 4 === 0) {
          await new Promise((resolve) => window.setTimeout(resolve, 0));
        }

        // Re-read current state to ensure we have the latest player after any
        // phase transitions that may have occurred during the chain capture.
        const stateForChain = hooks.getGameState();
        const chainPlayer = stateForChain.currentPlayer;

        const options = hooks.enumerateCaptureSegmentsFrom(chainPosition, chainPlayer);
        if (options.length === 0) {
          // RR-FIX-2026-01-12: Log warning if we're breaking out of chain capture
          // but phase is still chain_capture. This helps diagnose if phase transitions
          // aren't working correctly.
          if (stateForChain.currentPhase === 'chain_capture') {
            const allMoves = hooks.getValidMovesForCurrentPlayer();
            if (allMoves.length === 0) {
              console.warn(
                '[Sandbox AI] Chain capture stuck: no capture options and no valid moves. ' +
                  'Phase should have advanced but is still chain_capture.',
                {
                  boardType: stateForChain.boardType,
                  currentPlayer: chainPlayer,
                  chainPosition,
                  steps,
                }
              );
            }
          }
          break;
        }

        const nextSeg = options.reduce((best, current) => {
          const bx = best.landing.x;
          const by = best.landing.y;
          const bz = best.landing.z !== undefined ? best.landing.z : 0;
          const cx = current.landing.x;
          const cy = current.landing.y;
          const cz = current.landing.z !== undefined ? current.landing.z : 0;

          if (cx < bx) return current;
          if (cx > bx) return best;
          if (cy < by) return current;
          if (cy > by) return best;
          if (cz < bz) return current;
          if (cz > bz) return best;
          return best;
        }, options[0]);

        const stateForMove = hooks.getGameState();
        const continuationMoveNumber = stateForMove.history.length + 1;

        const continuationMove: Move = {
          id: '',
          type: 'continue_capture_segment',
          player: chainPlayer,
          from: chainPosition,
          captureTarget: nextSeg.target,
          to: nextSeg.landing,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: continuationMoveNumber,
        } as Move;

        await hooks.applyCanonicalMove(continuationMove);
        chainPosition = nextSeg.landing;
      }

      return;
    }

    if (selectedMove.type === 'skip_capture') {
      const stateForMove = hooks.getGameState();

      // Validate we're still in capture phase and it's still our turn
      if (
        stateForMove.currentPlayer !== selectedMove.player ||
        stateForMove.currentPhase !== 'capture'
      ) {
        return;
      }

      const moveNumber = stateForMove.history.length + 1;

      const skipMove: Move = {
        ...selectedMove,
        id: '',
        moveNumber,
        timestamp: new Date(),
      } as Move;

      await hooks.applyCanonicalMove(skipMove);

      lastAIMove = skipMove;
      hooks.setLastAIMove(lastAIMove);
      return;
    }

    if (normalizeLegacyMoveType(selectedMove.type) === 'move_stack') {
      const stateBeforeMove = hooks.getGameState();

      // Validate we're still in movement phase and it's still our turn
      if (
        stateBeforeMove.currentPlayer !== selectedMove.player ||
        stateBeforeMove.currentPhase !== 'movement'
      ) {
        return;
      }

      const moveNumber = stateBeforeMove.history.length + 1;

      const movementMove: Move = {
        ...selectedMove,
        id: '',
        moveNumber,
        timestamp: new Date(),
      } as Move;

      await hooks.applyCanonicalMove(movementMove);

      lastAIMove = movementMove;
      hooks.setLastAIMove(lastAIMove);
      return;
    }

    // Any other move types in movement phase are unexpected for the
    // sandbox AI; treat as a no-op so parity/debug harnesses can
    // highlight the mismatch.
    return;
  } finally {
    const afterStateForHistory = hooks.getGameState();
    const afterHashForHistory = hashGameState(afterStateForHistory);

    const stateUnchanged = beforeHashForHistory === afterHashForHistory;
    const samePlayer =
      debugPlayerNumber !== null && afterStateForHistory.currentPlayer === debugPlayerNumber;
    const samePhase =
      debugPhaseBefore !== null && afterStateForHistory.currentPhase === debugPhaseBefore;
    const stillActive = afterStateForHistory.gameStatus === 'active';

    // Stall detection: if state hash is unchanged AND player/phase are same, count as no-op.
    // If phase changed (e.g. forced_elimination -> ring_placement) or player changed,
    // it's progress even if hash somehow collided (unlikely) or if we want to be extra safe.
    if (stateUnchanged && samePlayer && samePhase && stillActive) {
      sandboxConsecutiveNoopAITurns += 1;
    } else {
      sandboxConsecutiveNoopAITurns = 0;
    }

    // Global safety net: if we observe a sustained sequence of no-op AI
    // turns for the same player while the game remains active, mark the
    // game as completed to avoid structural stalls.
    if (
      stateUnchanged &&
      samePlayer &&
      stillActive &&
      sandboxConsecutiveNoopAITurns >= SANDBOX_NOOP_STALL_THRESHOLD
    ) {
      const isActiveNoMoves = isANMState(afterStateForHistory);
      const allPlayersAI =
        afterStateForHistory.players.length > 0 &&
        afterStateForHistory.players.every((p) => p.type === 'ai');

      if (isActiveNoMoves) {
        // True ACTIVE_NO_MOVES state: the current player has no legal actions.
        // Run victory evaluation to determine if the game is actually over.
        const victoryResult = evaluateVictory(afterStateForHistory);

        // CRITICAL: Only mark game as completed if evaluateVictory confirms
        // the game is actually over. isANMState only checks the current player,
        // but another player may still have legal moves.
        if (victoryResult.isGameOver) {
          // Game is truly over - determine winner and mark completed
          const winner =
            victoryResult.winner !== undefined
              ? victoryResult.winner
              : applyStalemateLadder(afterStateForHistory);

          const stalled: GameState = {
            ...afterStateForHistory,
            gameStatus: 'completed',
            // Set terminal phase to 'game_over' and clear capture-specific
            // cursor state so completed sandbox games do not present as if
            // they were still mid-capture or awaiting a mandatory continuation.
            currentPhase: 'game_over',
            chainCapturePosition: undefined,
            mustMoveFromStackKey: undefined,
            winner,
          };
          hooks.setGameState(stalled);
          sandboxStallLoggingSuppressed = true;
        } else if (!sandboxStallLoggingSuppressed && SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED) {
          // ANM for current player but victory says game isn't over - another
          // player still has legal moves. Log this case but do NOT force
          // completion; let the turn/phase machinery advance to the next player.
          console.warn(
            '[Sandbox AI Stall Detector] isANMState=true but evaluateVictory.isGameOver=false; ' +
              'current player has no moves but game should continue',
            {
              boardType: afterStateForHistory.boardType,
              currentPlayer: afterStateForHistory.currentPlayer,
              currentPhase: afterStateForHistory.currentPhase,
              consecutiveNoopAITurns: sandboxConsecutiveNoopAITurns,
            }
          );
          // Suppress further logging for this stall situation but don't end game
        }
      } else if (!sandboxStallLoggingSuppressed && SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED) {
        // Hash-based stall window was reached but ANM(state) is false:
        // the active player still has some legal action according to
        // the global action summary. This indicates an internal AI or
        // host wiring bug; leave the game ACTIVE and emit a detailed
        // warning instead of force-completing a valid game.
        const summary = computeGlobalLegalActionsSummary(
          afterStateForHistory,
          afterStateForHistory.currentPlayer
        );
        console.warn(
          '[Sandbox AI Stall Detector] No-op window reached but ANM=false (valid moves exist); game left active',
          {
            boardType: afterStateForHistory.boardType,
            currentPlayer: afterStateForHistory.currentPlayer,
            currentPhase: afterStateForHistory.currentPhase,
            consecutiveNoopAITurns: sandboxConsecutiveNoopAITurns,
            allPlayersAI,
            globalActionsSummary: summary,
          }
        );
        sandboxStallLoggingSuppressed = true;
      }
    }

    if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && debugIsAiTurn) {
      const traceBuffer = getSandboxTraceBuffer();
      if (traceBuffer) {
        const turnEntry: SandboxAITurnTraceEntry = {
          kind: 'ai_turn',
          timestamp: Date.now(),
          boardType: debugBoardType ?? beforeStateForHistory.boardType,
          playerNumber: debugPlayerNumber,
          currentPhaseBefore: debugPhaseBefore ?? beforeStateForHistory.currentPhase,
          currentPhaseAfter: afterStateForHistory.currentPhase,
          gameStatusBefore: beforeStateForHistory.gameStatus,
          gameStatusAfter: afterStateForHistory.gameStatus,
          beforeHash: beforeHashForHistory,
          afterHash: afterHashForHistory,
          lastAIMoveType: lastAIMove ? lastAIMove.type : null,
          lastAIMovePlayer: lastAIMove ? lastAIMove.player : null,
          captureCount: debugCaptureCount ?? undefined,
          simpleMoveCount: debugSimpleMoveCount ?? undefined,
          placementCandidateCount: debugPlacementCandidateCount ?? undefined,
          forcedEliminationAttempted: debugForcedEliminationAttempted || undefined,
          forcedEliminationEliminated: debugForcedEliminationEliminated || undefined,
          consecutiveNoopAITurns: sandboxConsecutiveNoopAITurns || undefined,
          aiDecisionSource: debugAiDecisionSource,
          aiDifficultyRequested: debugAiDifficultyRequested ?? undefined,
          serviceAiType:
            debugAiDecisionSource === 'service' || debugAiDecisionSource === 'mismatch'
              ? (debugServiceAiType ?? undefined)
              : undefined,
          serviceDifficulty:
            debugAiDecisionSource === 'service' || debugAiDecisionSource === 'mismatch'
              ? (debugServiceDifficulty ?? undefined)
              : undefined,
          heuristicProfileId:
            debugAiDecisionSource === 'service' || debugAiDecisionSource === 'mismatch'
              ? (debugServiceHeuristicProfileId ?? undefined)
              : undefined,
          useNeuralNet:
            debugAiDecisionSource === 'service' || debugAiDecisionSource === 'mismatch'
              ? (debugServiceUseNeuralNet ?? undefined)
              : undefined,
          nnModelId:
            debugAiDecisionSource === 'service' || debugAiDecisionSource === 'mismatch'
              ? (debugServiceNnModelId ?? undefined)
              : undefined,
          nnCheckpoint:
            debugAiDecisionSource === 'service' || debugAiDecisionSource === 'mismatch'
              ? (debugServiceNnCheckpoint ?? undefined)
              : undefined,
          nnueCheckpoint:
            debugAiDecisionSource === 'service' || debugAiDecisionSource === 'mismatch'
              ? (debugServiceNnueCheckpoint ?? undefined)
              : undefined,
          thinkingTimeMs:
            debugAiDecisionSource === 'service' || debugAiDecisionSource === 'mismatch'
              ? (debugServiceThinkingTimeMs ?? undefined)
              : undefined,
          serviceError: debugServiceError ?? undefined,
        };

        traceBuffer.push(turnEntry);
        if (traceBuffer.length > MAX_SANDBOX_TRACE_ENTRIES) {
          traceBuffer.splice(0, traceBuffer.length - MAX_SANDBOX_TRACE_ENTRIES);
        }

        if (
          stateUnchanged &&
          samePlayer &&
          stillActive &&
          sandboxConsecutiveNoopAITurns >= SANDBOX_NOOP_STALL_THRESHOLD
        ) {
          const stallEntry: SandboxAITurnTraceEntry = {
            ...turnEntry,
            kind: 'stall',
          };
          traceBuffer.push(stallEntry);
          if (traceBuffer.length > MAX_SANDBOX_TRACE_ENTRIES) {
            traceBuffer.splice(0, traceBuffer.length - MAX_SANDBOX_TRACE_ENTRIES);
          }

          if (!sandboxStallLoggingSuppressed) {
            console.warn('[Sandbox AI Stall Detector] Detected potential AI stall', {
              boardType: turnEntry.boardType,
              playerNumber: turnEntry.playerNumber,
              currentPhase: turnEntry.currentPhaseAfter,
              consecutiveNoopAITurns: sandboxConsecutiveNoopAITurns,
            });
          }
        }
      }
    }

    if (
      debugIsAiTurn &&
      debugPlayerNumber !== null &&
      debugBoardType !== null &&
      debugAiDifficultyRequested !== null
    ) {
      recordSandboxAiDiagnostics({
        timestamp: Date.now(),
        gameId: beforeStateForHistory.id,
        boardType: debugBoardType,
        numPlayers: afterStateForHistory.players.length,
        playerNumber: debugPlayerNumber,
        requestedDifficulty: debugAiDifficultyRequested,
        source: debugAiDecisionSource,
        ...(debugAiDecisionSource === 'service' || debugAiDecisionSource === 'mismatch'
          ? {
              aiType: debugServiceAiType ?? undefined,
              difficulty: debugServiceDifficulty ?? undefined,
              heuristicProfileId: debugServiceHeuristicProfileId,
              useNeuralNet: debugServiceUseNeuralNet,
              nnModelId: debugServiceNnModelId,
              nnCheckpoint: debugServiceNnCheckpoint,
              nnueCheckpoint: debugServiceNnueCheckpoint,
              thinkingTimeMs: debugServiceThinkingTimeMs,
            }
          : {}),
        ...(debugServiceError ? { error: debugServiceError } : {}),
      });
    }

    // History entries for AI-driven actions are recorded by the canonical
    // move-applier (applyCanonicalMove / applyCanonicalMoveInternal) inside
    // ClientSandboxEngine. maybeRunAITurnSandbox is responsible only for
    // selecting logical Moves and delegating their application, so we do not
    // append additional history entries here.
  }
}
