import {
  BoardType,
  GameState,
  Move,
  Territory,
  Position,
  positionToString,
} from '../../src/shared/types/game';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { findMatchingBackendMove } from '../utils/moveMatching';
import { BoardManager } from '../../src/server/game/BoardManager';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { findDisconnectedRegions as findDisconnectedRegionsShared } from '../../src/shared/engine/territoryDetection';
import { findDisconnectedRegionsOnBoard } from '../../src/client/sandbox/sandboxTerritory';

// NOTE: Diagnostic seed-5 territory decision parity harness; currently skipped because
// its extensive console diagnostics can exceed Jest reporter limits. Core territory
// parity is covered by TerritoryDecisions.GameEngine_vs_Sandbox and related suites.
describe('Territory decision parity at seed 5 move 45', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 5;
  const MAX_STEPS = 60;

  function createSandboxEngineFromInitial(initial: GameState): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType: initial.boardType,
      numPlayers: initial.players.length,
      playerKinds: initial.players
        .slice()
        .sort((a, b) => a.playerNumber - b.playerNumber)
        .map((p) => p.type as 'human' | 'ai'),
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice(choice: any) {
        const options = ((choice as any).options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as any;
      },
    };

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler: handler,
      traceMode: true,
    });
    const engineAny: any = engine;
    engineAny.gameState = initial;
    return engine;
  }

  function canProcessRegionForSandboxAIView(
    state: GameState,
    region: Territory,
    playerNumber: number
  ): boolean {
    const regionPositionSet = new Set(region.spaces.map((pos) => positionToString(pos)));
    for (const stack of state.board.stacks.values()) {
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

  test('instrument backend vs sandbox territory state at sandbox moveNumber 45', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
    expect(trace.entries.length).toBeGreaterThan(0);

    // Historically this decision occurred at moveNumber 45 in the seed-5 trace.
    // After introducing explicit move-driven line/territory decisions, earlier
    // line-reward eliminations and region-processing moves may shift the absolute
    // moveNumber while preserving the same geometric/territory scenario. Rather
    // than pinning to a hard-coded moveNumber, locate the first explicit
    // process_territory_region decision for player 2 and treat that as the
    // canonical inspection point.
    const targetIndex = trace.entries.findIndex(
      (entry) => entry.action.type === 'process_territory_region' && entry.action.player === 2
    );
    expect(targetIndex).toBeGreaterThanOrEqual(0);

    const targetEntry = trace.entries[targetIndex];
    const targetMoveNumber = (targetEntry.action as Move).moveNumber;

    // Rebuild sandbox state before move 45
    const sandboxEngine = createSandboxEngineFromInitial(trace.initialState);
    for (const entry of trace.entries) {
      const move = entry.action as Move;
      if (move.moveNumber >= targetMoveNumber) {
        break;
      }
      await sandboxEngine.applyCanonicalMove(move);
    }
    const sandboxStateBefore = sandboxEngine.getGameState();

    // Rebuild backend state before move 45 using a replay similar to replayMovesOnBackend
    const backendEngine = createBackendEngineFromInitialState(trace.initialState);
    const backendMovesPrefix = trace.entries
      .map((e) => e.action as Move)
      .filter((m) => m.moveNumber < targetMoveNumber);

    for (const move of backendMovesPrefix) {
      const backendStateBeforeStep = backendEngine.getGameState();
      const backendValidMoves = backendEngine.getValidMoves(backendStateBeforeStep.currentPlayer);
      const matching = findMatchingBackendMove(move, backendValidMoves);

      if (!matching) {
        // eslint-disable-next-line no-console
        console.error('[TerritoryDecision.seed5Move45] backend prefix replay mismatch', {
          moveNumber: move.moveNumber,
          type: move.type,
          player: move.player,
          from: move.from,
          to: move.to,
          captureTarget: move.captureTarget,
          backendCurrentPlayer: backendStateBeforeStep.currentPlayer,
          backendCurrentPhase: backendStateBeforeStep.currentPhase,
          backendValidMovesCount: backendValidMoves.length,
        });
        throw new Error(
          `Backend prefix replay failed before moveNumber ${targetMoveNumber}; ` +
            `no matching backend move for sandbox move ` +
            JSON.stringify({
              moveNumber: move.moveNumber,
              type: move.type,
              player: move.player,
              from: move.from,
              to: move.to,
              captureTarget: move.captureTarget,
            })
        );
      }

      const { id, timestamp, moveNumber, ...payload } = matching as any;
      const result = await backendEngine.makeMove(
        payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );
      if (!result.success) {
        throw new Error(
          `Backend makeMove failed during prefix replay at backend moveNumber=${matching.moveNumber}: ${result.error}`
        );
      }
    }
    const backendStateBefore = (backendEngine as any).gameState as GameState;

    const sandboxBoard = sandboxStateBefore.board;
    const sandboxMovingPlayer = sandboxStateBefore.currentPlayer;

    const keyFromRegion = (spaces: Position[]) => spaces.map((p) => positionToString(p)).sort();

    // Detectors on sandbox board
    const sharedRegionsSandbox = findDisconnectedRegionsShared(sandboxBoard);
    const sandboxRegionsSandbox = findDisconnectedRegionsOnBoard(sandboxBoard);
    const boardManagerSandbox = new BoardManager(sandboxBoard.type);
    const backendViewRegionsOnSandbox = boardManagerSandbox.findDisconnectedRegions(
      sandboxBoard,
      sandboxMovingPlayer
    );

    const sharedKeysSandbox = sharedRegionsSandbox.map((r) => keyFromRegion(r.spaces));
    const sandboxKeysSandbox = sandboxRegionsSandbox.map((r) => keyFromRegion(r.spaces));
    const backendViewKeysSandbox = backendViewRegionsOnSandbox.map((r) => keyFromRegion(r.spaces));

    const sandboxAICanProcessFlags = sharedRegionsSandbox.map((r) =>
      canProcessRegionForSandboxAIView(sandboxStateBefore, r, sandboxMovingPlayer)
    );

    // Detectors and gating on backend board
    const backendBoard = backendStateBefore.board;
    const backendMovingPlayer = backendStateBefore.currentPlayer;

    const sharedRegionsBackend = findDisconnectedRegionsShared(backendBoard);
    const sandboxRegionsBackend = findDisconnectedRegionsOnBoard(backendBoard);
    const boardManagerBackend = new BoardManager(backendBoard.type);
    const backendRegionsBackend = boardManagerBackend.findDisconnectedRegions(
      backendBoard,
      backendMovingPlayer
    );

    const sharedKeysBackend = sharedRegionsBackend.map((r) => keyFromRegion(r.spaces));
    const sandboxKeysBackend = sandboxRegionsBackend.map((r) => keyFromRegion(r.spaces));
    const backendKeysBackend = backendRegionsBackend.map((r) => keyFromRegion(r.spaces));

    const backendEngineAny: any = backendEngine;
    const backendCanProcessFlags = backendRegionsBackend.map((r) =>
      backendEngineAny.canProcessDisconnectedRegion(r, backendMovingPlayer)
    );
    const pendingTerritoryFlag = backendEngineAny.pendingTerritorySelfElimination === true;

    const backendValidMovesAt45 = backendEngine.getValidMoves(backendMovingPlayer);
    const backendRegionMoves = backendValidMovesAt45.filter(
      (m) => m.type === 'process_territory_region'
    );
    const backendEliminationMoves = backendValidMovesAt45.filter(
      (m) => m.type === 'eliminate_rings_from_stack'
    );

    const sandboxEngineAny: any = sandboxEngine;
    const sandboxPendingFlag = sandboxEngineAny._pendingTerritorySelfElimination === true;
    const sandboxDecisionMoves: Move[] =
      sandboxEngineAny.getValidTerritoryProcessingMovesForCurrentPlayer() ?? [];
    const sandboxDecisionRegionMoves = sandboxDecisionMoves.filter(
      (m) => m.type === 'process_territory_region'
    );
    const sandboxDecisionEliminationMoves = sandboxDecisionMoves.filter(
      (m) => m.type === 'eliminate_rings_from_stack'
    );

    // eslint-disable-next-line no-console
    console.log('[TerritoryDecision.seed5Move45] diagnostic', {
      targetMoveNumber,
      targetAction: targetEntry.action,
      sandbox: {
        currentPlayer: sandboxStateBefore.currentPlayer,
        currentPhase: sandboxStateBefore.currentPhase,
        gameStatus: sandboxStateBefore.gameStatus,
        pendingTerritorySelfElimination: sandboxPendingFlag,
        sharedRegionKeys: sharedKeysSandbox,
        sandboxRegionKeys: sandboxKeysSandbox,
        backendViewRegionKeys: backendViewKeysSandbox,
        aiCanProcessFlags: sandboxAICanProcessFlags,
        decisionRegionMoves: sandboxDecisionRegionMoves.map((m) => ({
          id: m.id,
          player: m.player,
          disconnectedCount: (m as any).disconnectedRegions?.[0]?.spaces?.length ?? 0,
        })),
        decisionEliminationMoves: sandboxDecisionEliminationMoves.map((m) => ({
          to: m.to,
          player: m.player,
        })),
      },
      backend: {
        currentPlayer: backendStateBefore.currentPlayer,
        currentPhase: backendStateBefore.currentPhase,
        gameStatus: backendStateBefore.gameStatus,
        pendingTerritorySelfElimination: pendingTerritoryFlag,
        sharedRegionKeys: sharedKeysBackend,
        sandboxRegionKeys: sandboxKeysBackend,
        backendRegionKeys: backendKeysBackend,
        backendCanProcessFlags,
        validRegionMoves: backendRegionMoves.map((m) => ({
          id: m.id,
          player: m.player,
          disconnectedCount: (m as any).disconnectedRegions?.[0]?.spaces?.length ?? 0,
        })),
        validEliminationMoves: backendEliminationMoves.map((m) => ({
          to: m.to,
          player: m.player,
        })),
      },
    });

    // Soft expectations: both engines should be in territory_processing for player 2 at this point.
    expect(sandboxStateBefore.currentPlayer).toBe(2);
    expect(sandboxStateBefore.currentPhase).toBe('territory_processing');
    expect(backendStateBefore.currentPlayer).toBe(2);
    expect(backendStateBefore.currentPhase).toBe('territory_processing');
  });
});
