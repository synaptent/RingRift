import {
  GameState,
  Move,
  Position,
  positionToString,
  RingStack,
  BoardState,
  BOARD_CONFIGS,
} from '../../shared/types/game';
import { hashGameState } from '../../shared/engine/core';
import { chooseLocalMoveFromCandidates, LocalAIRng } from '../../shared/engine/localAIMoveSelection';
import {
  isSandboxAiCaptureDebugEnabled,
  isSandboxAiStallDiagnosticsEnabled,
} from '../../shared/utils/envFlags';

const SANDBOX_AI_CAPTURE_DEBUG_ENABLED = isSandboxAiCaptureDebugEnabled();
const SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED = isSandboxAiStallDiagnosticsEnabled();

const SANDBOX_NOOP_STALL_THRESHOLD = 5;
const MAX_SANDBOX_TRACE_ENTRIES = 2000;

// Module-level counter tracking consecutive AI turns that leave the
// sandbox GameState hash unchanged while the same AI player remains
// to move. Used as a structural stall detector in dev/test builds.
let sandboxConsecutiveNoopAITurns = 0;

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
  tryPlaceRings(position: Position, count: number): boolean;
  enumerateCaptureSegmentsFrom(
    from: Position,
    playerNumber: number
  ): Array<{ from: Position; target: Position; landing: Position }>;
  enumerateSimpleMovementLandings(
    playerNumber: number
  ): Array<{ fromKey: string; to: Position }>;
  maybeProcessForcedEliminationForCurrentPlayer(): boolean;
  handleMovementClick(position: Position): Promise<void>;
  appendHistoryEntry(before: GameState, action: Move): void;
  getGameState(): GameState;
  setGameState(state: GameState): void;
  setLastAIMove(move: Move | null): void;
  setSelectedStackKey(key: string | undefined): void;
  getMustMoveFromStackKey(): string | undefined;
  applyCanonicalMove(move: Move): Promise<void>;
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
    const current = gameState.players.find(
      (p) => p.playerNumber === gameState.currentPlayer
    );

    debugBoardType = gameState.boardType;
    debugPhaseBefore = gameState.currentPhase;
    debugPlayerNumber = gameState.currentPlayer;
    debugIsAiTurn = !!current && current.type === 'ai' && gameState.gameStatus === 'active';

    if (!current || current.type !== 'ai' || gameState.gameStatus !== 'active') {
      return;
    }

    // === Ring placement phase: canonical candidates + shared selector ===
    if (gameState.currentPhase === 'ring_placement') {
      const ringsInHand = current.ringsInHand ?? 0;
      if (ringsInHand <= 0) {
        // With no rings in hand, backend RuleEngine never exposes a
        // skip_placement move; treat this AI tick as a no-op and rely on
        // the surrounding controller/turn engine to advance phases.
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

      if (placementCandidates.length === 0) {
        // No legal placement under sandbox no-dead-placement. Only emit a
        // canonical skip_placement when backend RuleEngine would also
        // consider it legal: the player has rings in hand, controls at
        // least one stack, and has at least one legal move/capture from
        // some controlled stack (optional placement case).
        if (hasAnyActionFromStacks) {
          hooks.setGameState({
            ...gameState,
            currentPhase: 'movement',
          });

          lastAIMove = {
            id: '',
            type: 'skip_placement',
            player: current.playerNumber,
            from: undefined,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: gameState.history.length + 1,
          } as Move;
          hooks.setLastAIMove(lastAIMove);
        }

        // Otherwise placement is mandatory or state is structurally
        // blocked; do nothing this tick so parity harness can surface
        // any mismatch instead of us forging a skip_placement.
        return;
      }

      // Build canonical candidate moves for the shared local selection
      // policy. Each legal placement position becomes a place_ring move;
      // when optional placement is legal, we also include one or more
      // skip_placement candidates to approximate the ratio between
      // placement options and movement options available after a skip.
      const baseMoveNumber = gameState.history.length + 1;

      const placementMoves: Move[] = placementCandidates.map((pos, idx) =>
        ({
          id: '',
          type: 'place_ring',
          player: current.playerNumber,
          from: undefined,
          to: pos,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: baseMoveNumber + idx,
        } as Move)
      );

      const candidates: Move[] = [...placementMoves];

      if (hasAnyActionFromStacks) {
        // Approximate the number of movement options that would be
        // available if we skipped placement on this turn. We combine
        // all simple movements and capture segments from the current
        // board so that skip_placement is weighted against place_ring
        // by the relative counts.
        let totalMovementMovesForSkip = 0;

        for (const stack of playerStacksForSkip) {
          totalMovementMovesForSkip +=
            hooks.enumerateCaptureSegmentsFrom(stack.position, current.playerNumber).length;
        }

        const simpleMovesForSkip = hooks.enumerateSimpleMovementLandings(current.playerNumber);
        totalMovementMovesForSkip += simpleMovesForSkip.length;

        const skipMultiplicity = Math.max(1, totalMovementMovesForSkip);

        for (let i = 0; i < skipMultiplicity; i++) {
          candidates.push({
            id: '',
            type: 'skip_placement',
            player: current.playerNumber,
            from: undefined,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: baseMoveNumber + placementMoves.length + i,
          } as Move);
        }
      }

      const selected = chooseLocalMoveFromCandidates(
        current.playerNumber,
        gameState,
        candidates,
        rng
      );

      if (!selected) {
        return;
      }

      if (selected.type === 'skip_placement') {
        hooks.setGameState({
          ...gameState,
          currentPhase: 'movement',
        });

        lastAIMove = {
          id: '',
          type: 'skip_placement',
          player: current.playerNumber,
          from: undefined,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: gameState.history.length + 1,
        } as Move;
        hooks.setLastAIMove(lastAIMove);
        return;
      }

      if (selected.type !== 'place_ring' || !selected.to) {
        // Unexpected move type in ring_placement; treat as a no-op so
        // parity/debug harnesses can flag the structural mismatch.
        return;
      }

      const choice = selected.to;
      const board = gameState.board;
      const key = positionToString(choice);
      const existing = board.stacks.get(key);
      const isOccupied = !!existing && existing.rings.length > 0;

      // Compute the maximum number of rings we are allowed to place in this
      // action, mirroring backend RuleEngine semantics:
      //   - Per-player cap from BOARD_CONFIGS[boardType].ringsPerPlayer.
      //   - Limited by ringsInHand.
      //   - On empty cells we may place up to 3 rings at once.
      //   - On existing stacks we place at most 1 ring at a time.
      const boardConfig = BOARD_CONFIGS[gameState.boardType];
      const stacksForPlayer = hooks.getPlayerStacks(current.playerNumber, board);
      const ringsOnBoard = stacksForPlayer.reduce((sum, stack) => sum + stack.stackHeight, 0);

      const perPlayerCap = boardConfig.ringsPerPlayer;
      const remainingByCap = perPlayerCap - ringsOnBoard;
      const remainingBySupply = ringsInHand;
      const maxAvailable = Math.min(remainingByCap, remainingBySupply);

      if (maxAvailable <= 0) {
        return;
      }

      const maxPerPlacement = isOccupied ? 1 : Math.min(3, maxAvailable);

      const candidateCounts: number[] = [];
      for (let c = 1; c <= maxPerPlacement; c++) {
        candidateCounts.push(c);
      }

      for (let i = candidateCounts.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = candidateCounts[i];
        candidateCounts[i] = candidateCounts[j];
        candidateCounts[j] = tmp;
      }

      let placed = false;
      let effectiveCount = 1;

      for (const count of candidateCounts) {
        if (hooks.tryPlaceRings(choice, count)) {
          placed = true;
          effectiveCount = count;
          break;
        }
      }

      if (!placed) {
        if (hasAnyActionFromStacks) {
          hooks.setGameState({
            ...gameState,
            currentPhase: 'movement',
          });

          lastAIMove = {
            id: '',
            type: 'skip_placement',
            player: current.playerNumber,
            from: undefined,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: gameState.history.length + 1,
          } as Move;
          hooks.setLastAIMove(lastAIMove);
        }
        return;
      }

      lastAIMove = {
        id: '',
        type: 'place_ring',
        player: current.playerNumber,
        from: undefined,
        to: choice,
        placementCount: effectiveCount,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: gameState.history.length + 1,
      } as Move;
      hooks.setLastAIMove(lastAIMove);
      return;
    }

    // === Movement phase: canonical capture + movement candidates ===
    if (gameState.currentPhase !== 'movement') {
      return;
    }

    const playerNumber = current.playerNumber;

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

    // Enforce must-move semantics for simple movement as well: when a
    // placement has occurred this turn, only moves originating from the
    // placed/updated stack are eligible.
    if (mustMoveFromStackKey) {
      landingCandidates = landingCandidates.filter(
        (m) => m.fromKey === mustMoveFromStackKey
      );
    }

    const hasCaptures = captureSegments.length > 0;
    const hasMoves = landingCandidates.length > 0;

    debugCaptureCount = captureSegments.length;
    debugSimpleMoveCount = landingCandidates.length;

    if (!hasCaptures && !hasMoves) {
      const beforeElimState = hooks.getGameState();
      const beforeElimHash = hashGameState(beforeElimState);

      const eliminated = hooks.maybeProcessForcedEliminationForCurrentPlayer();
      debugForcedEliminationAttempted = true;
      debugForcedEliminationEliminated = eliminated;

      const afterElimState = hooks.getGameState();
      const afterElimHash = hashGameState(afterElimState);

      if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED) {
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

    // Build canonical Move[] candidates for captures and simple
    // movements, then let the shared helper choose which category to
    // draw from. Within each chosen category, we preserve the previous
    // deterministic tie-breaking (lexicographically smallest landing
    // position) to keep traces reproducible.
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

    const selectedMove = chooseLocalMoveFromCandidates(
      playerNumber,
      gameState,
      movementCandidates,
      rng
    );

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

        if (
          stateAfter.gameStatus !== 'active' ||
          stateAfter.currentPhase !== 'chain_capture'
        ) {
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
        debugPlayerNumber !== null &&
        afterStateForHistory.currentPlayer === debugPlayerNumber;
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

    // Only record a history entry when the AI actually produced a
    // canonical action and the sandbox state changed. Capture chains
    // are now recorded segment-by-segment via the movement engine
    // (handleCaptureSegmentApplied), so we intentionally skip
    // appending a separate high-level entry for capture segments
    // here to avoid double-counting.
    if (
      lastAIMove &&
      beforeHashForHistory !== afterHashForHistory &&
      lastAIMove.type !== 'overtaking_capture' &&
      lastAIMove.type !== 'continue_capture_segment' &&
      lastAIMove.type !== 'move_stack' &&
      lastAIMove.type !== 'move_ring'
    ) {
      hooks.appendHistoryEntry(beforeStateForHistory, lastAIMove);
    }
  }
}
