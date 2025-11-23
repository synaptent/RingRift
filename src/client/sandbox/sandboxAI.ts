import {
  GameState,
  Move,
  Position,
  positionToString,
  RingStack,
  BoardState,
  BOARD_CONFIGS,
  Territory,
} from '../../shared/types/game';
import { hashGameState } from '../../shared/engine/core';
import {
  LocalAIRng,
  chooseLocalMoveFromCandidates,
} from '../../shared/engine/localAIMoveSelection';
import {
  isSandboxAiCaptureDebugEnabled,
  isSandboxAiStallDiagnosticsEnabled,
  isSandboxAiParityModeEnabled,
} from '../../shared/utils/envFlags';
import { findAllLinesOnBoard } from './sandboxLines';
import { findDisconnectedRegions as findDisconnectedRegionsShared } from '../../shared/engine/territoryDetection';

const SANDBOX_AI_CAPTURE_DEBUG_ENABLED = isSandboxAiCaptureDebugEnabled();
const SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED = isSandboxAiStallDiagnosticsEnabled();

const SANDBOX_NOOP_STALL_THRESHOLD = 5;
const SANDBOX_NOOP_MAX_THRESHOLD = 10; // Stop execution after this many consecutive no-ops
const MAX_SANDBOX_TRACE_ENTRIES = 2000;

// Module-level counter tracking consecutive AI turns that leave the
// sandbox GameState hash unchanged while the same AI player remains
// to move. Used as a structural stall detector in dev/test builds.
let sandboxConsecutiveNoopAITurns = 0;
let sandboxStallLoggingSuppressed = false;

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
  captureCount?: number;
  simpleMoveCount?: number;
  placementCandidateCount?: number;
  forcedEliminationAttempted?: boolean;
  forcedEliminationEliminated?: boolean;
  consecutiveNoopAITurns?: number;
}

declare global {
  interface Window {
    __RINGRIFT_SANDBOX_TRACE__?: SandboxAITurnTraceEntry[];
  }
}

/**
 * Get (and lazily initialise) the in-browser sandbox AI trace buffer
 * used for debugging AI stalls. In non-browser or non-diagnostics
 * builds this returns null so the rest of the code can no-op.
 */
function getSandboxTraceBuffer(): SandboxAITurnTraceEntry[] | null {
  if (!SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED) {
    return null;
  }

  if (typeof window === 'undefined') {
    return null;
  }

  const anyWindow = window as any;
  if (!Array.isArray(anyWindow.__RINGRIFT_SANDBOX_TRACE__)) {
    anyWindow.__RINGRIFT_SANDBOX_TRACE__ = [];
  }

  return anyWindow.__RINGRIFT_SANDBOX_TRACE__ as SandboxAITurnTraceEntry[];
}

export interface SandboxAIHooks {
  getPlayerStacks(playerNumber: number, board: BoardState): RingStack[];
  hasAnyLegalMoveOrCaptureFrom(from: Position, playerNumber: number, board: BoardState): boolean;
  enumerateLegalRingPlacements(playerNumber: number): Position[];
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
}

function getLineDecisionMovesForSandboxAI(gameState: GameState): Move[] {
  const moves: Move[] = [];

  if (gameState.currentPhase !== 'line_processing') {
    return moves;
  }

  const playerNumber = gameState.currentPlayer;
  const boardType = gameState.boardType;
  const board = gameState.board;
  const requiredLength = BOARD_CONFIGS[boardType].lineLength;

  const allLines = findAllLinesOnBoard(
    boardType,
    board,
    (pos: Position) => {
      const config = BOARD_CONFIGS[boardType];
      if (boardType === 'hexagonal') {
        const radius = config.size - 1;
        const x = pos.x;
        const y = pos.y;
        const z = pos.z !== undefined ? pos.z : -x - y;
        const distance = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
        return distance <= radius;
      }
      return pos.x >= 0 && pos.x < config.size && pos.y >= 0 && pos.y < config.size;
    },
    (posStr: string) => {
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
    }
  );

  const playerLines = allLines.filter((line) => line.player === playerNumber);
  if (playerLines.length === 0) {
    return moves;
  }

  // One process_line move per player-owned line.
  playerLines.forEach((line, index) => {
    const lineKey = line.positions.map((p) => positionToString(p)).join('|');
    moves.push({
      id: `process-line-${index}-${lineKey}`,
      type: 'process_line',
      player: playerNumber,
      formedLines: [line],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: gameState.history.length + 1,
    } as Move);
  });

  // For overlength lines, also expose a choose_line_reward decision so the
  // AI can explicitly choose Option 1 vs Option 2 when available.
  const overlengthLines = playerLines.filter((line) => line.positions.length > requiredLength);
  overlengthLines.forEach((line, index) => {
    const lineKey = line.positions.map((p) => positionToString(p)).join('|');
    moves.push({
      id: `choose-line-reward-${index}-${lineKey}`,
      type: 'choose_line_reward',
      player: playerNumber,
      formedLines: [line],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: gameState.history.length + 1,
    } as Move);
  });

  return moves;
}

function canProcessRegionForSandboxAI(
  gameState: GameState,
  region: Territory,
  playerNumber: number
): boolean {
  const regionPositionSet = new Set(region.spaces.map((pos) => positionToString(pos)));

  for (const stack of gameState.board.stacks.values()) {
    if (stack.controllingPlayer !== playerNumber) {
      continue;
    }

    const key = positionToString(stack.position);
    if (!regionPositionSet.has(key)) {
      return true;
    }
  }

  return false;
}

function getTerritoryDecisionMovesForSandboxAI(
  gameState: GameState,
  hooks: SandboxAIHooks
): Move[] {
  const moves: Move[] = [];

  if (gameState.currentPhase !== 'territory_processing') {
    return moves;
  }

  const movingPlayer = gameState.currentPlayer;
  const board = gameState.board;

  const disconnected = findDisconnectedRegionsShared(board);
  const eligible: Territory[] = disconnected
    ? disconnected.filter((region) => canProcessRegionForSandboxAI(gameState, region, movingPlayer))
    : [];

  if (eligible.length > 0) {
    eligible.forEach((region, index) => {
      const representative = region.spaces[0];
      const regionKey = representative ? positionToString(representative) : `region-${index}`;
      moves.push({
        id: `process-region-${index}-${regionKey}`,
        type: 'process_territory_region',
        player: movingPlayer,
        disconnectedRegions: [region],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: gameState.history.length + 1,
      } as Move);
    });
    return moves;
  }

  // At this point, no eligible regions remain. Only surface explicit
  // elimination decisions when the engine reports a pending self-elimination
  // debt for this player, mirroring the backend
  // GameEngine.pendingTerritorySelfElimination flag.
  if (!hooks.hasPendingTerritorySelfElimination()) {
    return moves;
  }

  // Generate eliminate_rings_from_stack decisions for all controlled stacks.
  const playerStacks = hooks.getPlayerStacks(movingPlayer, board);
  if (playerStacks.length === 0) {
    return moves;
  }

  playerStacks.forEach((stack) => {
    const stackKey = positionToString(stack.position);
    moves.push({
      id: `eliminate-${stackKey}`,
      type: 'eliminate_rings_from_stack',
      player: movingPlayer,
      to: stack.position,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: gameState.history.length + 1,
    } as Move);
  });

  return moves;
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
    const beforeFilter = landingCandidates.length;
    if (beforeFilter > 0) {
      const first = landingCandidates[0];
      // eslint-disable-next-line no-console
      console.log('[Sandbox AI Debug] Before filter', {
        mustMoveFromStackKey,
        firstFromKey: first.fromKey,
        match: first.fromKey === mustMoveFromStackKey,
        allFromKeysLength: landingCandidates.length,
        has27: landingCandidates.some((m) => m.fromKey === '2,7'),
        mustMoveIs27: mustMoveFromStackKey === '2,7',
      });
    }
    landingCandidates = landingCandidates.filter((m) => {
      const match = m.fromKey === mustMoveFromStackKey;
      if (!match && m.fromKey === '2,7') {
        // eslint-disable-next-line no-console
        console.log('[Sandbox AI Debug] Filter mismatch:', {
          fromKey: m.fromKey,
          mustMoveFromStackKey,
          fromKeyCodes: m.fromKey.split('').map((c) => c.charCodeAt(0)),
          mustCodes: mustMoveFromStackKey.split('').map((c) => c.charCodeAt(0)),
        });
      }
      return match;
    });
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

export function selectSandboxMovementMove(
  gameState: GameState,
  candidates: Move[],
  rng: LocalAIRng,
  parityMode: boolean
): Move | null {
  const playerNumber = gameState.currentPlayer;

  if (parityMode) {
    // Parity mode: delegate directly to the shared selector so sandbox
    // movement/capture policy matches the backend local fallback.
    return chooseLocalMoveFromCandidates(playerNumber, gameState, candidates, rng);
  }

  // Default sandbox behaviour: currently identical to parity mode, but
  // structured so that a future heuristic policy can be introduced here
  // without affecting the parity path.
  return chooseLocalMoveFromCandidates(playerNumber, gameState, candidates, rng);
}

/**
 * Run a single AI turn in sandbox mode.
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
export async function maybeRunAITurnSandbox(
  hooks: SandboxAIHooks,
  rng: LocalAIRng = Math.random
): Promise<void> {
  // If we've exceeded the maximum consecutive no-op threshold, stop
  // executing AI turns to prevent infinite stalls and log spam.
  if (sandboxConsecutiveNoopAITurns >= SANDBOX_NOOP_MAX_THRESHOLD) {
    if (!sandboxStallLoggingSuppressed && SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED) {
      // eslint-disable-next-line no-console
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

    const parityMode = isSandboxAiParityModeEnabled();

    // === Ring placement phase: canonical candidates + shared selector ===
    if (gameState.currentPhase === 'ring_placement') {
      const ringsInHand = current.ringsInHand ?? 0;
      if (ringsInHand <= 0) {
        // With no rings in hand, ring_placement is not a real decision
        // phase under the rules: the player must either move from
        // existing stacks or, if completely blocked, undergo forced
        // elimination. Leaving this as a pure no-op would violate the
        // termination ladder and allow structural stalls when the
        // surrounding controller does not advance phases on its own.
        const boardForSkip = gameState.board;
        const playerStacksForSkip = hooks.getPlayerStacks(current.playerNumber, boardForSkip);
        const hasAnyActionFromStacks =
          playerStacksForSkip.length > 0 &&
          playerStacksForSkip.some((stack) =>
            hooks.hasAnyLegalMoveOrCaptureFrom(stack.position, current.playerNumber, boardForSkip)
          );

        if (hasAnyActionFromStacks) {
          // Legal moves exist from existing stacks: mirror backend
          // RuleEngine.validateSkipPlacement by issuing an explicit
          // skip_placement move so we always progress into the movement
          // phase rather than silently stalling.
          const stateForMove = hooks.getGameState();
          const moveNumber = stateForMove.history.length + 1;
          // For canonical skip_placement, no board coordinate is semantically
          // meaningful; use the shared sentinel {0,0} so sandbox traces align
          // with backend GameEngine and shared-engine fixtures.
          const sentinelTo: Position = { x: 0, y: 0 };

          const skipMove: Move = {
            id: '',
            type: 'skip_placement',
            player: current.playerNumber,
            from: undefined,
            to: sentinelTo,
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move;

          await hooks.applyCanonicalMove(skipMove);

          lastAIMove = skipMove;
          hooks.setLastAIMove(lastAIMove);
        } else {
          // No rings in hand and no legal moves/placements: this is a
          // true forced-elimination situation. Delegate to the shared
          // helper so we either eliminate a cap and advance to the next
          // player or, in degenerate cases, at least log a diagnostic
          // when state does not change.
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
            // eslint-disable-next-line no-console
            console.warn(
              '[Sandbox AI Stall Diagnostic] Ring-placement (no rings) forced elimination did not change state',
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
        }

        return;
      }

      // Mirror backend RuleEngine.validateSkipPlacement: skip is only
      // legal when placement is optional, i.e. the player both has
      // rings in hand and has at least one controlled stack with a
      // legal move/capture available.
      const boardForSkip = gameState.board;
      const playerStacksForSkip = hooks.getPlayerStacks(current.playerNumber, boardForSkip);
      const hasAnyActionFromStacks =
        playerStacksForSkip.length > 0 &&
        playerStacksForSkip.some((stack) =>
          hooks.hasAnyLegalMoveOrCaptureFrom(stack.position, current.playerNumber, boardForSkip)
        );

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
            hasAnyActionFromStacks,
            playerStacksCount: playerStacksForSkip.length,
          })
        );
      }

      // If no legal placement under sandbox no-dead-placement, we may still
      // skip placement when backend would also allow it; otherwise, the
      // state is structurally blocked.
      if (placementCandidates.length === 0) {
        if (hasAnyActionFromStacks) {
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
            from: undefined,
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
          // eslint-disable-next-line no-console
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
      // still skip placement if they have legal moves from existing stacks.
      // This mirrors the logic at lines 520-550 for the placementCandidates === 0 case.
      if (maxAvailableGlobal <= 0) {
        if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
          // eslint-disable-next-line no-console
          console.warn(
            '[Sandbox AI Debug] maxAvailableGlobal <= 0, checking for skip_placement option'
          );
        }

        if (hasAnyActionFromStacks) {
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
            from: undefined,
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
          // eslint-disable-next-line no-console
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
            from: undefined,
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

      if (hasAnyActionFromStacks) {
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
          // eslint-disable-next-line no-console
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

      const selected = chooseLocalMoveFromCandidates(
        current.playerNumber,
        gameState,
        candidates,
        rng
      );

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
          })
        );
      }

      if (!selected) {
        // eslint-disable-next-line no-console
        console.error(
          '[Sandbox AI] chooseLocalMoveFromCandidates returned null with',
          candidates.length,
          'candidates'
        );
        return;
      }

      const stateForMove = hooks.getGameState();
      const moveNumber = stateForMove.history.length + 1;

      let moveToApply: Move | null = null;

      if (selected.type === 'skip_placement') {
        moveToApply = {
          id: '',
          type: 'skip_placement',
          player: current.playerNumber,
          from: undefined,
          // Preserve the sentinel position chosen when we fabricated the
          // skip_placement candidate so traces match the Python engine.
          to: selected.to ?? ({ x: 0, y: 0 } as Position),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        } as Move;
      } else if (selected.type === 'place_ring') {
        // Removed the `&& selected.to` check that was causing stalls when selected.to
        // was somehow missing/corrupted. Instead, provide a defensive fallback.
        if (!selected.to) {
          // eslint-disable-next-line no-console
          console.error('[Sandbox AI] place_ring selected but to is missing:', selected);
          return;
        }
        moveToApply = {
          id: '',
          type: 'place_ring',
          player: current.playerNumber,
          from: undefined,
          to: selected.to,
          placementCount: selected.placementCount ?? 1,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        } as Move;
      } else {
        // Unexpected move type in ring_placement; log for debugging.
        // eslint-disable-next-line no-console
        console.error('[Sandbox AI] Unexpected move type in ring_placement:', selected.type);
        return;
      }

      await hooks.applyCanonicalMove(moveToApply);

      lastAIMove = moveToApply;
      hooks.setLastAIMove(lastAIMove);
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
        // process_line / choose_line_reward moves. This keeps the sandbox AI
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
          decisionCandidates = getLineDecisionMovesForSandboxAI(gameState);
        }
      } else {
        decisionCandidates = getTerritoryDecisionMovesForSandboxAI(gameState, hooks);
      }

      if (decisionCandidates.length === 0) {
        return;
      }

      const selectedDecision = chooseLocalMoveFromCandidates(
        gameState.currentPlayer,
        gameState,
        decisionCandidates,
        rng
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

    // === Movement phase: canonical capture + movement candidates ===
    if (gameState.currentPhase !== 'movement') {
      return;
    }

    const playerNumber = current.playerNumber;

    const { candidates: movementCandidates, debug: movementDebug } = buildSandboxMovementCandidates(
      gameState,
      hooks,
      rng
    );

    debugCaptureCount = movementDebug.captureCount;
    debugSimpleMoveCount = movementDebug.simpleMoveCount;

    if (movementDebug.captureCount === 0 && movementDebug.simpleMoveCount === 0) {
      const mustMoveFromStackKey = hooks.getMustMoveFromStackKey();
      let stacksForDebug = hooks.getPlayerStacks(playerNumber, gameState.board);
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
        }));
        // eslint-disable-next-line no-console
        console.log('[Sandbox AI Debug] No moves found', {
          phase: gameState.currentPhase,
          mustMoveFromStackKey,
          stacksCount: stacksForDebug.length,
          captureCount: movementDebug.captureCount,
          landingCount: movementDebug.simpleMoveCount,
          player: playerNumber,
          stackDetails,
        });
      }

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
          // eslint-disable-next-line no-console
          console.warn(
            '[Sandbox AI Stall Diagnostic] No captures/moves, forced elimination did not change state',
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
      }

      return;
    }

    const selectedMove = selectSandboxMovementMove(gameState, movementCandidates, rng, parityMode);

    if (!selectedMove) {
      return;
    }

    if (selectedMove.type === 'overtaking_capture') {
      const stateBeforeCapture = hooks.getGameState();
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

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const stateAfter = hooks.getGameState();

        if (stateAfter.gameStatus !== 'active' || stateAfter.currentPhase !== 'chain_capture') {
          break;
        }

        steps++;
        if (steps > MAX_CHAIN_STEPS) {
          // Defensive: avoid infinite loops if something goes wrong with
          // chain-capture phase transitions.
          // eslint-disable-next-line no-console
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

        const options = hooks.enumerateCaptureSegmentsFrom(chainPosition, playerNumber);
        if (options.length === 0) {
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
          player: playerNumber,
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

    if (selectedMove.type === 'move_stack' || selectedMove.type === 'move_ring') {
      const stateBeforeMove = hooks.getGameState();
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

    if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && debugIsAiTurn) {
      const stateUnchanged = beforeHashForHistory === afterHashForHistory;
      const samePlayer =
        debugPlayerNumber !== null && afterStateForHistory.currentPlayer === debugPlayerNumber;
      const stillActive = afterStateForHistory.gameStatus === 'active';

      if (stateUnchanged && samePlayer && stillActive) {
        sandboxConsecutiveNoopAITurns += 1;
      } else {
        sandboxConsecutiveNoopAITurns = 0;
      }

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
            // eslint-disable-next-line no-console
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

    // History entries for AI-driven actions are recorded by the canonical
    // move-applier (applyCanonicalMove / applyCanonicalMoveInternal) inside
    // ClientSandboxEngine. maybeRunAITurnSandbox is responsible only for
    // selecting logical Moves and delegating their application, so we do not
    // append additional history entries here.
  }
}
