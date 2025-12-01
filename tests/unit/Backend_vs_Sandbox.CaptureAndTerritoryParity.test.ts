import { GameEngine } from '../../src/server/game/GameEngine';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  BoardState,
  GameState,
  Move,
  Player,
  Position,
  TimeControl,
  Territory,
  LineInfo,
  positionToString,
} from '../../src/shared/types/game';
import {
  computeSMetric,
  computeTMetric,
  enumerateAllCaptureMoves,
  getChainCaptureContinuationInfo,
  enumerateProcessTerritoryRegionMoves,
  enumerateTerritoryEliminationMoves,
  getEffectiveLineLengthThreshold,
} from '../../src/shared/engine';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  addMarker,
  pos,
} from '../utils/fixtures';
import * as sharedLineDetection from '../../src/shared/engine/lineDetection';

/**
 * Backend vs Sandbox advanced-phase parity harness.
 *
 * HISTORICAL NOTE (Phase 3 Migration, 2025-12-01):
 * This test was originally designed to compare both hosts with orchestrator
 * disabled (legacy paths). As of Phase 3, the orchestrator adapter is
 * permanently enabled on ClientSandboxEngine and cannot be disabled.
 *
 * The test now compares:
 *  - GameEngine (backend host, orchestrator can still be disabled for tests)
 *  - ClientSandboxEngine (sandbox host, orchestrator permanently enabled)
 *
 * This still validates parity between backend and sandbox when processing
 * the same canonical Move sequences in advanced phases.
 *
 * Scope:
 *  - Pure capture chains (multi-step, with branching choices).
 *  - Territory disconnection and Q23-style self-elimination.
 *  - Combined line + territory long-turn ordering (line processed before region).
 *
 * Design notes:
 *  - These tests deliberately construct small, synthetic GameState scenarios
 *    using shared test fixtures, then inject clones of that state into:
 *      - GameEngine (backend host), and
 *      - ClientSandboxEngine (sandbox host).
 *  - Both hosts are then driven through the SAME canonical sequence of
 *    advanced-phase decisions using their public APIs:
 *      - backend: makeMove (move-driven decision phases enabled),
 *      - sandbox: applyCanonicalMove (orchestrator adapter enabled).
 *  - At each step we:
 *      1. Assert that legal decision surfaces match (capture / line / territory).
 *      2. Apply the same canonical Move to both hosts.
 *      3. Compare resulting board state, player statistics, and S/T metrics.
 *
 * These tests validate that both hosts produce identical results when
 * processing the same canonical Move sequences.
 */

// ──────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ──────────────────────────────────────────────────────────────────────────────

function cloneGameState(state: GameState): GameState {
  const board = state.board;
  const clonedBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };

  return {
    ...state,
    board: clonedBoard,
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
    history: [...state.history],
  };
}

interface HostPair {
  backend: GameEngine;
  backendAny: any;
  sandbox: ClientSandboxEngine;
  sandboxAny: any;
}

/**
 * Deterministic SandboxInteractionHandler used for parity tests.
 * - capture_direction: lexicographically smallest landing (x, then y).
 * - all other choices: first option.
 */
function createDeterministicSandboxHandler(): SandboxInteractionHandler {
  const handler: SandboxInteractionHandler = {
    async requestChoice<TChoice extends any>(choice: TChoice): Promise<any> {
      const anyChoice = choice as any;

      if (anyChoice.type === 'capture_direction') {
        const options = (anyChoice.options || []) as Array<{
          targetPosition: Position;
          landingPosition: Position;
        }>;
        if (options.length === 0) {
          throw new Error('SandboxInteractionHandler: no options for capture_direction');
        }

        let selected = options[0];
        for (const opt of options) {
          if (
            opt.landingPosition.x < selected.landingPosition.x ||
            (opt.landingPosition.x === selected.landingPosition.x &&
              opt.landingPosition.y < selected.landingPosition.y)
          ) {
            selected = opt;
          }
        }

        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption: selected,
        };
      }

      const optionsArray: any[] = (anyChoice.options as any[]) ?? [];
      const selectedOption = optionsArray.length > 0 ? optionsArray[0] : undefined;

      return {
        choiceId: anyChoice.id,
        playerNumber: anyChoice.playerNumber,
        choiceType: anyChoice.type,
        selectedOption,
      };
    },
  };

  return handler;
}

/**
 * Create backend and sandbox hosts that share the same cloned base GameState.
 *
 * - Backend:
 *   - move-driven decision phases enabled so line/territory decisions are
 *     expressed as canonical Moves (process_line, choose_line_reward,
 *     process_territory_region, eliminate_rings_from_stack).
 *
 * - Sandbox:
 *   - traceMode=true so line/territory decisions are surfaced as explicit
 *     canonical Moves instead of being auto-applied.
 *   - Orchestrator adapter is permanently enabled (Phase 3 migration).
 */
function createHostsFromBaseState(baseState: GameState): HostPair {
  const boardType: BoardType = baseState.boardType;
  const timeControl: TimeControl =
    (baseState.timeControl as TimeControl) ||
    ({ initialTime: 600, increment: 0, type: 'blitz' } as TimeControl);

  const backendPlayers: Player[] = baseState.players.map((p) => ({ ...p }));
  const backend = new GameEngine(
    'backend-advanced-phase-parity',
    boardType,
    backendPlayers,
    timeControl,
    false
  );
  // NOTE: Backend may still support disableOrchestratorAdapter() for test isolation.
  // For parity with sandbox (which now permanently uses orchestrator), we keep
  // it enabled on backend as well.
  backend.enableMoveDrivenDecisionPhases();
  const backendAny: any = backend;
  backendAny.gameState = cloneGameState(baseState);

  const sandboxConfig: SandboxConfig = {
    boardType,
    numPlayers: baseState.players.length,
    playerKinds: baseState.players.map((p) => p.type as 'human' | 'ai'),
  };
  const sandboxHandler = createDeterministicSandboxHandler();
  const sandbox = new ClientSandboxEngine({
    config: sandboxConfig,
    interactionHandler: sandboxHandler,
    traceMode: true,
  });
  // NOTE: As of Phase 3 migration, orchestrator adapter is permanently enabled
  // on ClientSandboxEngine. The disableOrchestratorAdapter() method no longer exists.
  const sandboxAny: any = sandbox;
  sandboxAny.gameState = cloneGameState(baseState);

  return { backend, backendAny, sandbox, sandboxAny };
}

interface NormalizedMove {
  type: Move['type'];
  player: number;
  from?: string;
  to?: string;
  captureTarget?: string;
  lineKey?: string;
  collapsedKey?: string;
  regionKey?: string;
}

/**
 * Normalise a Move into a parity-comparison signature.
 * We intentionally ignore ids, timestamps, and moveNumber.
 */
function normalizeMove(m: Move): NormalizedMove {
  const from = m.from ? positionToString(m.from) : undefined;
  const to = m.to ? positionToString(m.to) : undefined;
  const captureTarget = m.captureTarget ? positionToString(m.captureTarget) : undefined;

  let lineKey: string | undefined;
  if (m.formedLines && m.formedLines.length > 0) {
    const line = m.formedLines[0];
    const parts = line.positions.map((p) => positionToString(p)).sort();
    lineKey = `${line.player}:${parts.join('|')}`;
  }

  let collapsedKey: string | undefined;
  if (m.collapsedMarkers && m.collapsedMarkers.length > 0) {
    const parts = m.collapsedMarkers.map((p) => positionToString(p)).sort();
    collapsedKey = parts.join('|');
  }

  let regionKey: string | undefined;
  if (m.disconnectedRegions && m.disconnectedRegions.length > 0) {
    const region = m.disconnectedRegions[0];
    const parts = region.spaces.map((p) => positionToString(p)).sort();
    regionKey = `${region.controllingPlayer}:${parts.join('|')}`;
  }

  return {
    type: m.type,
    player: m.player,
    from,
    to,
    captureTarget,
    lineKey,
    collapsedKey,
    regionKey,
  };
}

function sortNormalizedMoves(arr: NormalizedMove[]): NormalizedMove[] {
  return [...arr].sort((a, b) => {
    const sa = JSON.stringify(a);
    const sb = JSON.stringify(b);
    return sa.localeCompare(sb);
  });
}

function expectMoveSetsEqual(
  backendMoves: Move[],
  sandboxMoves: Move[],
  label: string,
  filter?: (m: Move) => boolean
): void {
  const b = filter ? backendMoves.filter(filter) : backendMoves;
  const s = filter ? sandboxMoves.filter(filter) : sandboxMoves;

  const normB = sortNormalizedMoves(b.map(normalizeMove));
  const normS = sortNormalizedMoves(s.map(normalizeMove));

  try {
    expect(normS).toEqual(normB);
  } catch (err) {
    console.error('[Backend_vs_Sandbox.CaptureAndTerritoryParity] move surface divergence', {
      label,
      backendMoves: normB,
      sandboxMoves: normS,
    });
    throw err;
  }
}

/**
 * Select a canonical Move from a list by lexicographically sorting its
 * normalised representation. This keeps scenarios deterministic without
 * depending on host-specific ordering.
 */
function selectCanonicalMove(moves: Move[]): Move {
  if (moves.length === 0) {
    throw new Error('selectCanonicalMove: no moves available');
  }
  const withNorm = moves.map((m) => ({ move: m, sig: JSON.stringify(normalizeMove(m)) }));
  withNorm.sort((a, b) => a.sig.localeCompare(b.sig));
  return withNorm[0].move;
}

/**
 * Find a Move in `candidates` that normalises to the same signature as
 * `reference`. Returns undefined when no match exists.
 */
function findMatchingMove(reference: Move, candidates: Move[]): Move | undefined {
  const refSig = JSON.stringify(normalizeMove(reference));
  return candidates.find((m) => JSON.stringify(normalizeMove(m)) === refSig);
}

/**
 * Compare backend and sandbox GameStates for advanced-phase parity.
 *
 * - Board geometry: stack positions, owners, heights, caps.
 * - Collapsed territory ownership.
 * - Player-level territorySpaces and eliminatedRings.
 * - totalRingsEliminated.
 * - S/T metrics (computeSMetric / computeTMetric).
 */
function summarizeStateForLog(state: GameState) {
  const board = state.board;

  return {
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
    gameStatus: state.gameStatus,
    stacks: Array.from(board.stacks.entries()),
    collapsedSpaces: Array.from(board.collapsedSpaces.entries()),
    territories: Array.from(board.territories.entries()),
    formedLines: board.formedLines,
    eliminatedRings: board.eliminatedRings,
    players: state.players.map((p) => ({
      playerNumber: p.playerNumber,
      ringsInHand: p.ringsInHand,
      eliminatedRings: p.eliminatedRings,
      territorySpaces: p.territorySpaces,
    })),
    totalRingsEliminated: state.totalRingsEliminated,
    S: computeSMetric(state),
    T: computeTMetric(state),
  };
}

function expectStateParity(backendState: GameState, sandboxState: GameState, label: string): void {
  try {
    const bBoard = backendState.board;
    const sBoard = sandboxState.board;

    const bStackEntries = Array.from(bBoard.stacks.entries()).sort((a, b) =>
      a[0].localeCompare(b[0])
    );
    const sStackEntries = Array.from(sBoard.stacks.entries()).sort((a, b) =>
      a[0].localeCompare(b[0])
    );
    expect(sStackEntries.map(([k]) => k)).toEqual(bStackEntries.map(([k]) => k));

    bStackEntries.forEach(([key, stack], idx) => {
      const other = sStackEntries[idx][1];
      expect(other).toBeDefined();
      if (!other) return;
      expect(other.controllingPlayer).toBe(stack.controllingPlayer);
      expect(other.stackHeight).toBe(stack.stackHeight);
      expect(other.capHeight).toBe(stack.capHeight);
    });

    const bCollapsed = Array.from(bBoard.collapsedSpaces.entries()).sort((a, b) =>
      a[0].localeCompare(b[0])
    );
    const sCollapsed = Array.from(sBoard.collapsedSpaces.entries()).sort((a, b) =>
      a[0].localeCompare(b[0])
    );
    expect(sCollapsed).toEqual(bCollapsed);

    expect(sandboxState.players.length).toBe(backendState.players.length);
    backendState.players.forEach((bp) => {
      const sp = sandboxState.players.find((p) => p.playerNumber === bp.playerNumber);
      expect(sp).toBeDefined();
      if (!sp) return;
      expect(sp.territorySpaces).toBe(bp.territorySpaces);
      expect(sp.eliminatedRings).toBe(bp.eliminatedRings);
      expect(sp.ringsInHand).toBe(bp.ringsInHand);
    });

    expect(sBoard.eliminatedRings).toEqual(bBoard.eliminatedRings);
    expect(sandboxState.totalRingsEliminated).toBe(backendState.totalRingsEliminated);

    const sBackend = computeSMetric(backendState);
    const tBackend = computeTMetric(backendState);
    const sSandbox = computeSMetric(sandboxState);
    const tSandbox = computeTMetric(sandboxState);

    expect({ label, S: sSandbox, T: tSandbox }).toEqual({ label, S: sBackend, T: tBackend });
  } catch (err) {
    console.error('[Backend_vs_Sandbox.CaptureAndTerritoryParity] state divergence', {
      label,
      backend: summarizeStateForLog(backendState),
      sandbox: summarizeStateForLog(sandboxState),
    });
    throw err;
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Scenario 1: Pure capture chain (orthogonal branching choice)
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Build a square8 capture-chain scenario inspired by the orthogonal choice
 * configuration used in GameEngine.chainCaptureChoiceIntegration tests:
 *
 *  - Player 1 stack at (3,3) height 2 (attacker).
 *  - Player 2 at (3,4) height 1 (first target).
 *  - Player 3 at (4,5) height 1 (one continuation).
 *  - Player 4 at (2,5) height 1 (other continuation).
 */
function buildCaptureChainBaseState(): GameState {
  const boardType: BoardType = 'square8';
  const board = createTestBoard(boardType);

  const players: Player[] = [
    createTestPlayer(1),
    createTestPlayer(2),
    createTestPlayer(3),
    createTestPlayer(4),
  ];

  const state = createTestGameState({
    boardType,
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'movement',
    totalRingsEliminated: 0,
  });

  const red: Position = pos(3, 3);
  const blue: Position = pos(3, 4);
  const green: Position = pos(4, 5);
  const yellow: Position = pos(2, 5);

  addStack(board, red, 1, 2);
  addStack(board, blue, 2, 1);
  addStack(board, green, 3, 1);
  addStack(board, yellow, 4, 1);

  return state;
}

// ──────────────────────────────────────────────────────────────────────────────
// Scenario 2: Territory disconnection + Q23-style self-elimination
//   RulesMatrix anchor: Rules_12_2_Q23_mini_region_square8_numeric_invariant
// ──────────────────────────────────────────────────────────────────────────────
function buildTerritoryQ23BaseState(): {
  state: GameState;
  region: Territory;
  outsidePos: Position;
} {
  const boardType: BoardType = 'square8';
  const board = createTestBoard(boardType);

  const players: Player[] = [createTestPlayer(1), createTestPlayer(2)];

  const state = createTestGameState({
    boardType,
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'territory_processing',
    totalRingsEliminated: 0,
  });

  const regionSpaces: Position[] = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];

  // Victim stacks for player 2 in the region (one ring each).
  for (const p of regionSpaces) {
    addStack(board, p, 2, 1);
  }

  // Outside stack for player 1 to satisfy self-elimination prerequisite (height 3).
  const outsidePos = pos(0, 0);
  addStack(board, outsidePos, 1, 3);

  // Border markers for player 1 around the region, mirroring the
  // territoryProcessing.shared Q23-style mini-region fixture so that
  // disconnected-region detection can discover this region without any
  // testOverrideRegions.
  const borderCoords: Array<[number, number]> = [];
  for (let x = 1; x <= 4; x++) {
    borderCoords.push([x, 1]);
    borderCoords.push([x, 4]);
  }
  for (let y = 2; y <= 3; y++) {
    borderCoords.push([1, y]);
    borderCoords.push([4, y]);
  }
  borderCoords.forEach(([x, y]) => addMarker(board, pos(x, y), 1));

  const region: Territory = {
    spaces: regionSpaces,
    controllingPlayer: 1,
    isDisconnected: true,
  };

  return { state, region, outsidePos };
}

// ──────────────────────────────────────────────────────────────────────────────
// Scenario 3: Combined line + territory long turn
//   RulesMatrix anchor: Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_square8
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Combined line + territory scenario on square8:
 *
 *  - Overlength horizontal line for player 1 on row y=0:
 *      markers at (0,0)..(4,0) (represented via a synthetic LineInfo stub).
 *  - Single-cell disconnected region at (5,5) with a player-2 stack.
 *  - Player 1 stack at (7,7) height 2 used for both line reward elimination
 *    (in a full Option 1 variant) and Q23-style territory self-elimination.
 *
 * We stub shared line detection via sharedLineDetection.findAllLines so that
 * both backend and sandbox see the same synthetic line for decision helpers,
 * without relying on marker geometry.
 */
function buildLineAndTerritoryBaseState(): {
  state: GameState;
  lineInfo: LineInfo;
  territoryRegion: Territory;
  outsidePos: Position;
} {
  const boardType: BoardType = 'square8';
  const board = createTestBoard(boardType);
  const players: Player[] = [createTestPlayer(1), createTestPlayer(2)];
  const state = createTestGameState({
    boardType,
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'line_processing',
    totalRingsEliminated: 0,
  });

  const requiredLength = getEffectiveLineLengthThreshold(
    boardType,
    players.length,
    state.rulesOptions
  );
  const linePositions: Position[] = [];
  for (let i = 0; i < requiredLength + 1; i++) {
    linePositions.push(pos(i, 0));
  }

  const lineInfo: LineInfo = {
    positions: linePositions,
    player: 1,
    length: linePositions.length,
    direction: { x: 1, y: 0 },
  };

  // Territory region: single P2 stack at (5,5).
  const regionPos = pos(5, 5);
  addStack(board, regionPos, 2, 1);
  const territoryRegion: Territory = {
    spaces: [regionPos],
    controllingPlayer: 1,
    isDisconnected: true,
  };

  // Outside P1 stack (height 2) at (7,7).
  const outsidePos = pos(7, 7);
  addStack(board, outsidePos, 1, 2);

  // Border markers for player 1 around the single-cell region so that
  // shared territory detection treats this as a disconnected region
  // with a controlling border, mirroring the Q23-style fixtures.
  const borderOffsets: Array<[number, number]> = [
    [0, -1],
    [0, 1],
    [-1, 0],
    [1, 0],
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1],
  ];

  borderOffsets.forEach(([dx, dy]) => {
    addMarker(board, pos(regionPos.x + dx, regionPos.y + dy), 1);
  });

  return { state, lineInfo, territoryRegion, outsidePos };
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

describe('Backend vs Sandbox advanced-phase parity – capture, line, territory', () => {
  test('line_processing getValidMoves parity – overlength line order/reward decisions (square8)', async () => {
    const { state: baseState, lineInfo } = buildLineAndTerritoryBaseState();

    // Stub shared line detection so both hosts see the same synthetic
    // overlength line, independent of actual marker layout.
    const findAllLinesSpy = jest
      .spyOn(sharedLineDetection, 'findAllLines')
      .mockImplementation(() => [lineInfo]);

    try {
      const { backend, backendAny, sandbox, sandboxAny } = createHostsFromBaseState(baseState);
      backend.enableMoveDrivenDecisionPhases();

      // Ensure both hosts start from identical cloned state.
      expectStateParity(
        backendAny.gameState as GameState,
        sandboxAny.gameState as GameState,
        'line-processing-initial'
      );

      const currentPlayer = 1;

      const backendLineMoves = backend
        .getValidMoves(currentPlayer)
        .filter((m) => m.type === 'process_line' || m.type === 'choose_line_reward');

      const sandboxLineMoves: Move[] = (
        sandboxAny.getValidLineProcessingMovesForCurrentPlayer
          ? (sandboxAny.getValidLineProcessingMovesForCurrentPlayer() as Move[])
          : []
      ) as Move[];

      expect(backendLineMoves.length).toBeGreaterThan(0);
      expect(sandboxLineMoves.length).toBeGreaterThan(0);

      expectMoveSetsEqual(
        backendLineMoves,
        sandboxLineMoves,
        'line_processing getValidMoves parity (process_line/choose_line_reward)'
      );
    } finally {
      findAllLinesSpy.mockRestore();
    }
  });

  test('chain_capture getValidMoves parity – continuation surface from shared chain state', async () => {
    const baseState = buildCaptureChainBaseState();
    const { backend, backendAny, sandbox, sandboxAny } = createHostsFromBaseState(baseState);

    const currentPlayer = 1;

    // STEP 1: Initial capture from the base state so that a chain is available.
    const backendMoves0 = backend.getValidMoves(currentPlayer);
    const sandboxMoves0 = sandbox.getValidMoves(currentPlayer);

    const isCapture = (m: Move) =>
      m.type === 'overtaking_capture' || m.type === 'continue_capture_segment';

    const backendCaptures0 = backendMoves0.filter(isCapture);
    expect(backendCaptures0.length).toBeGreaterThan(0);

    const firstCaptureBackend = selectCanonicalMove(backendCaptures0);
    const firstCaptureSandbox =
      findMatchingMove(firstCaptureBackend, sandboxMoves0.filter(isCapture)) || firstCaptureBackend;

    const { id: _id0, timestamp: _ts0, moveNumber: _mn0, ...payload0 } = firstCaptureBackend as any;
    const res0 = await backend.makeMove(payload0 as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>);
    expect(res0.success).toBe(true);

    await sandbox.applyCanonicalMove(firstCaptureSandbox);

    const backendState = backendAny.gameState as GameState;
    const sandboxState = sandboxAny.gameState as GameState;

    expect(backendState.currentPhase).toBe('chain_capture');
    expect(sandboxState.currentPhase).toBe('chain_capture');

    const landing = firstCaptureBackend.to as Position;

    // STEP 2: Compare getValidMoves surfaces for chain continuations.
    const backendChainMoves = backend
      .getValidMoves(currentPlayer)
      .filter((m) => m.type === 'continue_capture_segment');
    const sandboxChainMoves = sandbox
      .getValidMoves(currentPlayer)
      .filter((m) => m.type === 'continue_capture_segment');

    expectMoveSetsEqual(
      backendChainMoves,
      sandboxChainMoves,
      'chain_capture getValidMoves surface after first segment'
    );

    // Additionally, compare the shared aggregate continuation surfaces from
    // the same landing position to ensure both hosts feed getValidMoves()
    // from the same canonical capture core.
    const infoBackend = getChainCaptureContinuationInfo(backendState, currentPlayer, landing);
    const infoSandbox = getChainCaptureContinuationInfo(sandboxState, currentPlayer, landing);

    expectMoveSetsEqual(
      infoBackend.availableContinuations,
      infoSandbox.availableContinuations,
      'shared aggregate continuation surface after first segment'
    );
  });

  test('pure capture chain parity – orthogonal branching scenario', async () => {
    const baseState = buildCaptureChainBaseState();
    const { backend, sandbox } = createHostsFromBaseState(baseState);

    const currentPlayer = 1;

    // STEP 1: Initial capture surface in the movement phase.
    const backendMoves0 = backend.getValidMoves(currentPlayer);
    const sandboxMoves0 = sandbox.getValidMoves(currentPlayer);

    const isCapture = (m: Move) =>
      m.type === 'overtaking_capture' || m.type === 'continue_capture_segment';

    expectMoveSetsEqual(backendMoves0, sandboxMoves0, 'initial capture surface', isCapture);

    const backendCaptures0 = backendMoves0.filter(isCapture);
    expect(backendCaptures0.length).toBeGreaterThan(0);

    const firstCaptureBackend = selectCanonicalMove(backendCaptures0);
    const firstCaptureSandbox =
      findMatchingMove(firstCaptureBackend, sandboxMoves0.filter(isCapture)) || firstCaptureBackend;

    // Apply first capture segment via host APIs.
    const { id: _id0, timestamp: _ts0, moveNumber: _mn0, ...payload0 } = firstCaptureBackend as any;
    const res0 = await backend.makeMove(payload0 as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>);
    expect(res0.success).toBe(true);

    await sandbox.applyCanonicalMove(firstCaptureSandbox);

    let backendState = backend.getGameState();
    let sandboxState = sandbox.getGameState();

    expectStateParity(backendState, sandboxState, 'after first capture segment');

    // STEP 2+: Chain continuation via shared capture aggregate.
    let landing = firstCaptureBackend.to as Position;
    let infoBackend = getChainCaptureContinuationInfo(backendState, currentPlayer, landing);
    let infoSandbox = getChainCaptureContinuationInfo(sandboxState, currentPlayer, landing);

    expectMoveSetsEqual(
      infoBackend.availableContinuations,
      infoSandbox.availableContinuations,
      'chain continuation surface after first segment'
    );

    let segmentIndex = 1;
    const MAX_SEGMENTS = 4;

    while (infoBackend.mustContinue && infoBackend.availableContinuations.length > 0) {
      expect(segmentIndex).toBeLessThanOrEqual(MAX_SEGMENTS);

      const contBackend = selectCanonicalMove(infoBackend.availableContinuations);
      const contSandbox =
        findMatchingMove(contBackend, infoSandbox.availableContinuations) || contBackend;

      const { id: _cid, timestamp: _cts, moveNumber: _cmn, ...payloadC } = contBackend as any;
      const resC = await backend.makeMove(
        payloadC as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );
      expect(resC.success).toBe(true);

      await sandbox.applyCanonicalMove(contSandbox);

      backendState = backend.getGameState();
      sandboxState = sandbox.getGameState();
      expectStateParity(backendState, sandboxState, `after capture segment ${segmentIndex + 1}`);

      landing = contBackend.to as Position;
      infoBackend = getChainCaptureContinuationInfo(backendState, currentPlayer, landing);
      infoSandbox = getChainCaptureContinuationInfo(sandboxState, currentPlayer, landing);

      expectMoveSetsEqual(
        infoBackend.availableContinuations,
        infoSandbox.availableContinuations,
        `chain continuation surface after segment ${segmentIndex + 1}`
      );

      segmentIndex += 1;
    }

    // Ensure no further captures are available for the capturing player and
    // both hosts expose identical final capture surfaces.
    const backendCapturesFinal = enumerateAllCaptureMoves(backendState, currentPlayer);
    const sandboxCapturesFinal = enumerateAllCaptureMoves(sandboxState, currentPlayer);
    expectMoveSetsEqual(
      backendCapturesFinal,
      sandboxCapturesFinal,
      'final capture surface after chain exhaustion',
      isCapture
    );
  });

  test('territory disconnection + Q23-style self-elimination parity (square8 mini-region)', async () => {
    const { state: baseState, region, outsidePos } = buildTerritoryQ23BaseState();
    const { backend, backendAny, sandbox, sandboxAny } = createHostsFromBaseState(baseState);

    const currentPlayer = 1;

    // Sanity: backend and sandbox start from identical states.
    expectStateParity(
      backendAny.gameState as GameState,
      sandboxAny.gameState as GameState,
      'initial'
    );

    // STEP 1: Enumerate region-processing decisions via shared helper and
    // ensure both hosts surface compatible process_territory_region moves.
    const backendEnumRegions = enumerateProcessTerritoryRegionMoves(
      backend.getGameState(),
      currentPlayer
    );
    const sandboxEnumRegions = enumerateProcessTerritoryRegionMoves(
      sandbox.getGameState(),
      currentPlayer
    );
    expect(backendEnumRegions.length).toBe(1);
    expectMoveSetsEqual(
      backendEnumRegions,
      sandboxEnumRegions,
      'region enumeration via shared helper'
    );

    const canonicalRegionMove = selectCanonicalMove(backendEnumRegions);

    // Hosts should both expose this region decision via getValidMoves in the
    // territory_processing phase.
    const backendSurface = backend
      .getValidMoves(currentPlayer)
      .filter((m) => m.type === 'process_territory_region');
    const sandboxSurface = sandbox
      .getValidMoves(currentPlayer)
      .filter((m) => m.type === 'process_territory_region');

    expectMoveSetsEqual(
      backendSurface,
      sandboxSurface,
      'host region-processing surfaces (process_territory_region)'
    );

    const backendRegionMove =
      findMatchingMove(canonicalRegionMove, backendSurface) || canonicalRegionMove;
    const sandboxRegionMove =
      findMatchingMove(canonicalRegionMove, sandboxSurface) || canonicalRegionMove;

    const {
      id: _ridB,
      timestamp: _rtsB,
      moveNumber: _rmnB,
      ...payloadRegion
    } = backendRegionMove as any;
    const resRegion = await backend.makeMove(
      payloadRegion as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
    );
    expect(resRegion.success).toBe(true);

    await sandbox.applyCanonicalMove(sandboxRegionMove);

    let backendState = backend.getGameState();
    let sandboxState = sandbox.getGameState();
    expectStateParity(backendState, sandboxState, 'after process_territory_region');

    // STEP 2: Enumerate self-elimination decisions via shared helper and
    // ensure both hosts expose identical eliminate_rings_from_stack surfaces.
    const backendEnumElims = enumerateTerritoryEliminationMoves(backendState, currentPlayer);
    const sandboxEnumElims = enumerateTerritoryEliminationMoves(sandboxState, currentPlayer);
    expect(backendEnumElims.length).toBeGreaterThan(0);
    expectMoveSetsEqual(
      backendEnumElims,
      sandboxEnumElims,
      'elimination enumeration via shared helper'
    );

    const canonicalElim = selectCanonicalMove(backendEnumElims);

    const backendElimSurface = backend
      .getValidMoves(currentPlayer)
      .filter((m) => m.type === 'eliminate_rings_from_stack');
    const sandboxElimSurface = sandbox
      .getValidMoves(currentPlayer)
      .filter((m) => m.type === 'eliminate_rings_from_stack');

    expectMoveSetsEqual(
      backendElimSurface,
      sandboxElimSurface,
      'host elimination decision surfaces (eliminate_rings_from_stack)'
    );

    const backendElimMove = findMatchingMove(canonicalElim, backendElimSurface) || canonicalElim;
    const sandboxElimMove = findMatchingMove(canonicalElim, sandboxElimSurface) || canonicalElim;

    const {
      id: _eidB,
      timestamp: _etsB,
      moveNumber: _emnB,
      ...payloadElim
    } = backendElimMove as any;
    const resElim = await backend.makeMove(
      payloadElim as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
    );
    expect(resElim.success).toBe(true);

    await sandbox.applyCanonicalMove(sandboxElimMove);

    backendState = backend.getGameState();
    sandboxState = sandbox.getGameState();
    expectStateParity(backendState, sandboxState, 'after territory self-elimination');

    // Additional numeric invariants: region stacks eliminated + one
    // self-elimination from the outside P1 stack.
    const backendBoard = backendState.board;
    const backendP1 = backendState.players.find((p) => p.playerNumber === 1)!;
    const backendP2 = backendState.players.find((p) => p.playerNumber === 2)!;

    // All region stacks for player 2 eliminated.
    const regionKeys = new Set(region.spaces.map((p) => positionToString(p)));
    const remainingRegionStacks = Array.from(backendBoard.stacks.keys()).filter((k) =>
      regionKeys.has(k)
    );
    expect(remainingRegionStacks.length).toBe(0);

    // Outside stack either reduced by at least 1 ring or removed entirely.
    const outsideKey = positionToString(outsidePos);
    const outsideStack = backendBoard.stacks.get(outsideKey);
    if (outsideStack) {
      expect(outsideStack.stackHeight).toBeLessThan(3);
    }

    // Player 1 should have at least region.rings + 1 self-elimination credited.
    const ringsEliminatedFromRegion = region.spaces.length; // 1 ring per victim stack
    expect(backendP1.eliminatedRings).toBeGreaterThanOrEqual(ringsEliminatedFromRegion + 1);
    // Player 2 has no eliminations credited in this Q23-style scenario.
    expect(backendP2.eliminatedRings).toBe(0);
  });

  test('combined line + territory parity (line then region + self-elimination, square8)', async () => {
    const {
      state: baseState,
      lineInfo,
      territoryRegion,
      outsidePos,
    } = buildLineAndTerritoryBaseState();

    // Stub shared line detection so both hosts see the same synthetic line,
    // independent of actual marker layout.
    const findAllLinesSpy = jest
      .spyOn(sharedLineDetection, 'findAllLines')
      .mockImplementation(() => [lineInfo]);

    try {
      const { backend, backendAny, sandbox, sandboxAny } = createHostsFromBaseState(baseState);
      backend.enableMoveDrivenDecisionPhases();

      // Ensure both hosts start from identical cloned state.
      expectStateParity(
        backendAny.gameState as GameState,
        sandboxAny.gameState as GameState,
        'combined-scenario-initial'
      );

      const currentPlayer = 1;
      const requiredLength = getEffectiveLineLengthThreshold(
        baseState.boardType,
        baseState.players.length,
        baseState.rulesOptions
      );

      // STEP 1: Enumerate canonical line-processing decisions for player 1.
      const backendLineMoves = backend
        .getValidMoves(currentPlayer)
        .filter((m) => m.type === 'process_line' || m.type === 'choose_line_reward');
      const sandboxLineMoves: Move[] = (
        sandboxAny.getValidLineProcessingMovesForCurrentPlayer
          ? sandboxAny.getValidLineProcessingMovesForCurrentPlayer()
          : []
      ) as Move[];

      expect(backendLineMoves.length).toBeGreaterThan(0);
      expect(sandboxLineMoves.length).toBeGreaterThan(0);

      expectMoveSetsEqual(
        backendLineMoves,
        sandboxLineMoves,
        'line-processing decision surfaces (process_line/choose_line_reward)'
      );

      // Choose the "Option 2" minimum-collapse reward for the overlength line
      // (collapsedMarkers length == requiredLength).
      const backendRewardMoves = backendLineMoves.filter((m) => m.type === 'choose_line_reward');
      const minCollapseBackend = backendRewardMoves.find(
        (m) => m.collapsedMarkers && m.collapsedMarkers.length === requiredLength
      );
      expect(minCollapseBackend).toBeDefined();

      const minCollapseSandbox =
        findMatchingMove(minCollapseBackend as Move, sandboxLineMoves) || minCollapseBackend!;

      // Apply the canonical choose_line_reward Move to both hosts.
      const {
        id: _lidB,
        timestamp: _ltsB,
        moveNumber: _lmnB,
        ...payloadLine
      } = minCollapseBackend as any;
      const resLine = await backend.makeMove(
        payloadLine as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );
      expect(resLine.success).toBe(true);

      await sandbox.applyCanonicalMove(minCollapseSandbox as Move);

      const backendState = backend.getGameState();
      const sandboxState = sandbox.getGameState();
      expectStateParity(backendState, sandboxState, 'after choose_line_reward (min collapse)');

      // STEP 2: Territory-processing decision surface after line resolution.
      // Use the shared helper for enumeration to confirm backend and sandbox
      // see the same candidate process_territory_region Moves, then ensure
      // that each host exposes those decisions through its own
      // getValidMoves-based surface for the interactive territory_processing
      // phase.
      const backendRegionEnum = enumerateProcessTerritoryRegionMoves(backendState, currentPlayer);
      const sandboxRegionEnum = enumerateProcessTerritoryRegionMoves(sandboxState, currentPlayer);
      expect(backendRegionEnum.length).toBeGreaterThan(0);
      expectMoveSetsEqual(
        backendRegionEnum,
        sandboxRegionEnum,
        'combined scenario region enumeration via shared helper'
      );

      const backendRegionSurface = backend
        .getValidMoves(currentPlayer)
        .filter((m) => m.type === 'process_territory_region');
      const sandboxRegionSurface = sandbox
        .getValidMoves(currentPlayer)
        .filter((m) => m.type === 'process_territory_region');

      expectMoveSetsEqual(
        backendRegionSurface,
        sandboxRegionSurface,
        'combined scenario region-processing surfaces'
      );
    } finally {
      findAllLinesSpy.mockRestore();
    }
  });
});
