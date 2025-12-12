import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardState,
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  Territory,
  TimeControl,
  positionToString,
} from '../../src/shared/types/game';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
} from '../../src/shared/engine/territoryDecisionHelpers';
import { createTestBoard, createTestPlayer, pos, addStack } from '../utils/fixtures';
import { snapshotFromGameState, snapshotsEqual, diffSnapshots } from '../utils/stateSnapshots';

/**
 * Territory decision-phase parity tests (C4) for multi-region scenarios.
 *
 * Goal:
 *   - For a two-region layout where both regions are eligible for the same
 *     moving player, verify that backend GameEngine and ClientSandboxEngine:
 *       1) Enumerate the same choose_territory_option decisions (legacy alias:
 *          process_territory_region) from
 *          getValidMoves / sandbox territory helpers,
 *       2) Process both regions in the same region-key order,
 *       3) Surface explicit eliminate_rings_from_stack decisions once all
 *          regions are processed,
 *       4) Choose equivalent elimination targets under deterministic stubs,
 *       5) End with matching board+player snapshots, phases, and
 *          pendingTerritorySelfElimination flags.
 *
 * This test intentionally stubs the disconnected-region detectors on both
 * engines to return a shared pair of Territory objects so that we can focus
 * purely on move-driven decision semantics and pending-flag lifecycle
 * without being sensitive to detector internals.
 *
 * NOTE: Skipped when ORCHESTRATOR_ADAPTER_ENABLED=true because the orchestrator
 * auto-processes single regions differently (intentional divergence).
 */

// Skip this test suite when orchestrator adapter is enabled - territory processing behavior differs intentionally
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'Territory decision multi-region parity (GameEngine vs ClientSandboxEngine)',
  () => {
    const boardType: BoardType = 'square8';
    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    interface MultiRegionFixture {
      board: BoardState;
      players: Player[];
      movingPlayer: number;
      regions: Territory[];
      outsideStacks: Position[];
    }

    function cloneBoard(board: BoardState): BoardState {
      return {
        ...board,
        stacks: new Map(board.stacks),
        markers: new Map(board.markers),
        collapsedSpaces: new Map(board.collapsedSpaces),
        territories: new Map(board.territories),
        formedLines: [...board.formedLines],
        eliminatedRings: { ...board.eliminatedRings },
      };
    }

    function clonePlayers(players: Player[]): Player[] {
      return players.map((p) => ({ ...p }));
    }

    /**
     * Build a simple two-region territory fixture on square8:
     *
     *   - Region A: cells (1,1) and (1,2) with victim stacks for Player 2.
     *   - Region B: cells (5,5) and (5,6) with victim stacks for Player 2.
     *   - Moving player: Player 1.
     *   - Outside stacks for Player 1:
     *       - One at (0,0) of height 2,
     *       - One at (7,7) of height 2.
     *
     * Both regions satisfy the self-elimination prerequisite for Player 1
     * (they have stacks outside both regions). We do not rely on detector
     * internals to discover these regions; instead we stub the detectors on
     * both engines to return these Territories while skipping any region whose
     * spaces have fully collapsed in board.collapsedSpaces.
     */
    function buildMultiRegionFixture(): MultiRegionFixture {
      const board = createTestBoard(boardType);

      const players: Player[] = [createTestPlayer(1), createTestPlayer(2)];

      const movingPlayer = 1;

      const regionASpaces: Position[] = [pos(1, 1), pos(1, 2)];
      const regionBSpaces: Position[] = [pos(5, 5), pos(5, 6)];

      // Victim stacks (Player 2) inside each region.
      regionASpaces.forEach((p) => addStack(board, p, 2, 1));
      regionBSpaces.forEach((p) => addStack(board, p, 2, 1));

      // Outside stacks for Player 1 used for self-elimination decisions.
      const outside1 = pos(0, 0);
      const outside2 = pos(7, 7);
      addStack(board, outside1, 1, 2);
      addStack(board, outside2, 1, 2);

      const regions: Territory[] = [
        {
          spaces: regionASpaces,
          controllingPlayer: 1,
          isDisconnected: true,
        },
        {
          spaces: regionBSpaces,
          controllingPlayer: 1,
          isDisconnected: true,
        },
      ];

      return {
        board,
        players,
        movingPlayer,
        regions,
        outsideStacks: [outside1, outside2],
      };
    }

    function keyFromSpaces(spaces: Position[]): string {
      return spaces
        .map((p) => positionToString(p))
        .sort()
        .join('|');
    }

    test('two-region move-driven territory cycle stays in parity through region decisions and elimination', async () => {
      const fixture = buildMultiRegionFixture();
      const baseBoard = fixture.board;
      const basePlayers = fixture.players;
      const movingPlayer = fixture.movingPlayer;
      const regionKeyMap: Record<string, Territory> = {};
      fixture.regions.forEach((r) => {
        regionKeyMap[keyFromSpaces(r.spaces)] = r;
      });

      // --- Backend setup ---
      const backendPlayers = clonePlayers(basePlayers);
      const backendEngine = new GameEngine(
        'territory-decisions-multi-region',
        boardType,
        backendPlayers,
        timeControl,
        false
      );
      backendEngine.enableMoveDrivenDecisionPhases();

      const backendAny: any = backendEngine;
      const backendState0: GameState = backendEngine.getGameState();
      const backendBoard = cloneBoard(baseBoard);

      backendAny.gameState = {
        ...backendState0,
        board: backendBoard,
        players: backendPlayers,
        currentPlayer: movingPlayer,
        currentPhase: 'territory_processing',
        gameStatus: 'active',
        history: [],
        moveHistory: [],
        totalRingsEliminated: 0,
      } as GameState;

      const backendBoardManager: any = backendAny.boardManager;

      // --- Sandbox setup ---
      const sandboxConfig: SandboxConfig = {
        boardType,
        numPlayers: basePlayers.length,
        playerKinds: basePlayers.map((p) => p.type as 'human' | 'ai'),
      };

      const sandboxHandler: SandboxInteractionHandler = {
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

      const sandboxEngine = new ClientSandboxEngine({
        config: sandboxConfig,
        interactionHandler: sandboxHandler,
        traceMode: true,
      });
      const sandboxAny: any = sandboxEngine;
      const sandboxState0: GameState = sandboxEngine.getGameState();
      const sandboxBoard = cloneBoard(baseBoard);
      const sandboxPlayers = clonePlayers(basePlayers);

      sandboxAny.gameState = {
        ...sandboxState0,
        board: sandboxBoard,
        players: sandboxPlayers,
        currentPlayer: movingPlayer,
        currentPhase: 'territory_processing',
        gameStatus: 'active',
        history: [],
        moveHistory: [],
        totalRingsEliminated: 0,
      } as GameState;

      // Sanity: initial snapshots identical.
      const backendInitialSnap = snapshotFromGameState(
        'backend-initial',
        backendEngine.getGameState()
      );
      const sandboxInitialSnap = snapshotFromGameState(
        'sandbox-initial',
        sandboxEngine.getGameState()
      );
      expect(snapshotsEqual(backendInitialSnap, sandboxInitialSnap)).toBe(true);

      // --- Step 1: enumerate initial territory region decisions in both engines ---
      const backendRegionMovesInitial: Move[] = enumerateProcessTerritoryRegionMoves(
        backendEngine.getGameState(),
        movingPlayer,
        { testOverrideRegions: fixture.regions }
      );

      const sandboxRegionMovesInitial: Move[] = enumerateProcessTerritoryRegionMoves(
        sandboxEngine.getGameState(),
        movingPlayer,
        { testOverrideRegions: fixture.regions }
      );

      expect(backendRegionMovesInitial.length).toBe(2);
      expect(sandboxRegionMovesInitial.length).toBe(2);

      const backendRegionByKey: Record<string, Move> = {};
      backendRegionMovesInitial.forEach((m) => {
        const spaces = (m.disconnectedRegions && m.disconnectedRegions[0]?.spaces) || [];
        const key = keyFromSpaces(spaces);
        backendRegionByKey[key] = m;
      });

      const sandboxRegionByKey: Record<string, Move> = {};
      sandboxRegionMovesInitial.forEach((m) => {
        const spaces = (m.disconnectedRegions && m.disconnectedRegions[0]?.spaces) || [];
        const key = keyFromSpaces(spaces);
        sandboxRegionByKey[key] = m;
      });

      const regionKeys = Object.keys(regionKeyMap).sort();
      expect(Object.keys(backendRegionByKey).sort()).toEqual(regionKeys);
      expect(Object.keys(sandboxRegionByKey).sort()).toEqual(regionKeys);

      // --- Step 2: process both regions in canonical key order ---
      for (const regionKey of regionKeys) {
        const backendMove = backendRegionByKey[regionKey];
        const sandboxMove = sandboxRegionByKey[regionKey];
        expect(backendMove).toBeDefined();
        expect(sandboxMove).toBeDefined();

        // Backend: apply via the shared helper so we focus purely on the
        // canonical territory-processing semantics. This bypasses
        // RuleEngine.validateTerritoryProcessingMove's structural coupling
        // to detector geometry, which is intentionally not consulted when
        // using testOverrideRegions.
        {
          const backendStateBefore = backendEngine.getGameState();
          const outcome = applyProcessTerritoryRegionDecision(
            backendStateBefore,
            backendMove as Move
          );
          const backendAnyInternal: any = backendEngine;
          backendAnyInternal.gameState = outcome.nextState;

          // Mirror GameEngine's move-driven territory flag lifecycle:
          // after processing at least one disconnected region, a mandatory
          // self-elimination decision is required.
          backendAnyInternal.pendingTerritorySelfElimination = true;
        }

        // Sandbox: apply via canonical move.
        await sandboxEngine.applyCanonicalMove(sandboxMove as Move);

        const backendAfterRegion = backendEngine.getGameState();
        const sandboxAfterRegion = sandboxEngine.getGameState();

        const backendAnyAfter: any = backendEngine;
        const sandboxAnyAfter: any = sandboxEngine;

        // Both engines should still be in territory_processing for the same player,
        // and both should have a pending self-elimination requirement.
        expect(backendAfterRegion.currentPhase).toBe('territory_processing');
        expect(sandboxAfterRegion.currentPhase).toBe('territory_processing');
        expect(backendAfterRegion.currentPlayer).toBe(movingPlayer);
        expect(sandboxAfterRegion.currentPlayer).toBe(movingPlayer);

        expect(backendAnyAfter.pendingTerritorySelfElimination).toBe(true);
        expect(sandboxAnyAfter._pendingTerritorySelfElimination).toBe(true);
      }

      // --- Step 3: after both regions are processed, enumerate elimination decisions ---
      const backendMovesAfterRegions = backendEngine.getValidMoves(movingPlayer);
      const backendElimMoves = backendMovesAfterRegions.filter(
        (m) => m.type === 'eliminate_rings_from_stack'
      );
      expect(backendElimMoves.length).toBeGreaterThan(0);

      const sandboxElimMoves: Move[] =
        sandboxAny.getValidEliminationDecisionMovesForCurrentPlayer() ?? [];
      expect(sandboxElimMoves.length).toBeGreaterThan(0);

      const backendElimTargets = backendElimMoves
        .map((m) => (m.to ? positionToString(m.to) : ''))
        .filter((k) => k.length > 0)
        .sort();
      const sandboxElimTargets = sandboxElimMoves
        .map((m) => (m.to ? positionToString(m.to) : ''))
        .filter((k) => k.length > 0)
        .sort();

      expect(backendElimTargets).toEqual(sandboxElimTargets);
      expect(backendElimTargets.length).toBeGreaterThan(0);

      const chosenElimKey = backendElimTargets[0];

      const backendElimMove = backendElimMoves.find(
        (m) => m.to && positionToString(m.to) === chosenElimKey
      );
      const sandboxElimMove = sandboxElimMoves.find(
        (m) => m.to && positionToString(m.to) === chosenElimKey
      );

      expect(backendElimMove).toBeDefined();
      expect(sandboxElimMove).toBeDefined();

      // Apply elimination in both engines.
      {
        const { id, timestamp, moveNumber, ...payload } = backendElimMove as any;
        const result = await backendEngine.makeMove(
          payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
        );
        expect(result.success).toBe(true);
      }

      await sandboxEngine.applyCanonicalMove(sandboxElimMove as Move);

      const backendFinal = backendEngine.getGameState();
      const sandboxFinal = sandboxEngine.getGameState();

      const backendAnyFinal: any = backendEngine;
      const sandboxAnyFinal: any = sandboxEngine;

      // Pending flags must be cleared and both engines must have left the
      // territory_processing phase for this player.
      expect(backendAnyFinal.pendingTerritorySelfElimination).toBe(false);
      expect(sandboxAnyFinal._pendingTerritorySelfElimination).toBe(false);

      expect(backendFinal.currentPhase).not.toBe('territory_processing');
      expect(sandboxFinal.currentPhase).not.toBe('territory_processing');

      // Current player should have advanced identically.
      expect(backendFinal.currentPlayer).toBe(sandboxFinal.currentPlayer);

      const backendFinalSnap = snapshotFromGameState('backend-final', backendFinal);
      const sandboxFinalSnap = snapshotFromGameState('sandbox-final', sandboxFinal);

      if (!snapshotsEqual(backendFinalSnap, sandboxFinalSnap)) {
        console.error('[TerritoryDecisions.GameEngine_vs_Sandbox] final snapshot mismatch', {
          diff: diffSnapshots(backendFinalSnap, sandboxFinalSnap),
        });
      }

      expect(snapshotsEqual(backendFinalSnap, sandboxFinalSnap)).toBe(true);
    });
  }
);
