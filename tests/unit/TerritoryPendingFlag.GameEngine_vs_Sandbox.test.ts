import {
  BoardState,
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  Territory,
} from '../../src/shared/types/game';
import { BoardManager } from '../../src/server/game/BoardManager';
import { GameEngine } from '../../src/server/game/GameEngine';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  snapshotFromBoardAndPlayers,
  snapshotsEqual,
  diffSnapshots,
} from '../utils/stateSnapshots';

/**
 * Focused parity test for pending territory self-elimination flags and
 * territory-processing phase transitions (C4).
 *
 * Scenario:
 *   - A single disconnected region exists for player 1, satisfying the
 *     self-elimination prerequisite (stacks both inside and outside).
 *   - We apply a single process_territory_region decision move in both
 *     engines using their respective core helpers.
 *   - We then apply a single eliminate_rings_from_stack decision move
 *     from a stack outside the region.
 *
 * Expectations:
 *   - After process_territory_region:
 *       - Backend and sandbox board+player snapshots match.
 *       - Backend pendingTerritorySelfElimination === true.
 *       - Sandbox _pendingTerritorySelfElimination === true.
 *       - Both engines are still in territory_processing for player 1.
 *   - After eliminate_rings_from_stack:
 *       - Backend and sandbox board+player snapshots still match.
 *       - Both pending flags are cleared.
 *       - Phases have advanced out of territory_processing.
 *
 * NOTE: Skipped when ORCHESTRATOR_ADAPTER_ENABLED=true because the orchestrator
 * auto-processes single regions differently (intentional divergence).
 */

// Skip this test suite when orchestrator adapter is enabled - territory processing behavior differs intentionally
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'Territory pending self-elimination flag & phase parity (backend vs sandbox)',
  () => {
    const boardType: BoardType = 'square8';

    function makeDummyPlayers(): Player[] {
      return [
        {
          id: 'p1',
          username: 'P1',
          type: 'ai',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 0,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'P2',
          type: 'ai',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 0,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];
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

    /**
     * Reuse the same style of fixture as TerritoryCore.GameEngine_vs_Sandbox:
     *
     * - regionSpaces: a small 2-cell region.
     * - borderMarkers: a short band just beyond the region.
     * - stacks:
     *   - inside region: stacks for both players,
     *   - outside region: a player-1 stack that will serve as the source
     *     of self-elimination after processing the region.
     */
    function buildRegionFixture() {
      const bm = new BoardManager(boardType);
      const baseBoard = bm.createBoard();

      const regionSpaces: Position[] = [
        { x: 2, y: 2 },
        { x: 3, y: 2 },
      ];

      const borderMarkers: Position[] = [
        { x: 2, y: 3 },
        { x: 3, y: 3 },
      ];

      const regionStackP1 = 1;
      const regionStackP2 = 2;

      // Stacks inside region
      bm.setStack(
        regionSpaces[0],
        {
          position: regionSpaces[0],
          rings: [regionStackP1, regionStackP1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: regionStackP1,
        },
        baseBoard
      );

      bm.setStack(
        regionSpaces[1],
        {
          position: regionSpaces[1],
          rings: [regionStackP2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: regionStackP2,
        },
        baseBoard
      );

      // Outside stack for P1 (eligible self-elimination source).
      const outsidePos: Position = { x: 0, y: 0 };
      bm.setStack(
        outsidePos,
        {
          position: outsidePos,
          rings: [regionStackP1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: regionStackP1,
        },
        baseBoard
      );

      // Border markers owned by P1.
      for (const pos of borderMarkers) {
        bm.setMarker(pos, regionStackP1, baseBoard);
      }

      const players = makeDummyPlayers();
      const movingPlayer = 1;

      const region: Territory = {
        spaces: regionSpaces,
        controllingPlayer: 0,
        isDisconnected: true,
      };

      return {
        baseBoard,
        players,
        region,
        regionSpaces,
        movingPlayer,
        outsidePos,
      };
    }

    test('pending flag and elimination semantics match across territory decision cycle', async () => {
      const { baseBoard, players, region, regionSpaces, movingPlayer, outsidePos } =
        buildRegionFixture();

      // --- Sandbox path: process_territory_region then eliminate_rings_from_stack ---
      const sandboxConfig: SandboxConfig = {
        boardType,
        numPlayers: players.length,
        playerKinds: players.map((p) => p.type as 'human' | 'ai'),
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
      const sandboxPlayers = players.map((p) => ({ ...p }));

      sandboxAny.gameState = {
        ...sandboxState0,
        board: sandboxBoard,
        players: sandboxPlayers,
        currentPlayer: movingPlayer,
        currentPhase: 'territory_processing',
      } as GameState;

      const regionMoveSandbox: Move = {
        id: 'process-region-test',
        type: 'process_territory_region',
        player: movingPlayer,
        disconnectedRegions: [region],
        timestamp: new Date(0),
        thinkTime: 0,
        moveNumber: 1,
        to: regionSpaces[0],
      } as Move;

      await sandboxEngine.applyCanonicalMove(regionMoveSandbox);

      const sandboxAfterRegion: GameState = sandboxEngine.getGameState();
      const sandboxRegionSnap = snapshotFromBoardAndPlayers(
        'sandbox-after-region',
        sandboxAfterRegion.board,
        sandboxAfterRegion.players
      );

      const sandboxPendingRegion = sandboxAny._pendingTerritorySelfElimination === true;
      expect(sandboxPendingRegion).toBe(true);
      expect(sandboxAfterRegion.currentPhase).toBe('territory_processing');
      expect(sandboxAfterRegion.currentPlayer).toBe(movingPlayer);

      // --- Backend path: applyDecisionMove(process_territory_region) via core helper ---
      const timeControl = { initialTime: 0, increment: 0, type: 'rapid' as const };
      const backendEngine = new GameEngine(
        'territory-pending-flag-test',
        boardType,
        players.map((p) => ({ ...p })),
        timeControl,
        false
      );
      backendEngine.enableMoveDrivenDecisionPhases();

      const backendAny: any = backendEngine;
      const backendState0: GameState = backendEngine.getGameState();
      const backendBoard = cloneBoard(baseBoard);
      const backendPlayers = players.map((p) => ({ ...p }));

      backendAny.gameState = {
        ...backendState0,
        board: backendBoard,
        players: backendPlayers,
        currentPlayer: movingPlayer,
        currentPhase: 'territory_processing',
      } as GameState;

      const regionMoveBackend: Move = {
        id: 'process-region-test',
        type: 'process_territory_region',
        player: movingPlayer,
        disconnectedRegions: [region],
        timestamp: new Date(0),
        thinkTime: 0,
        moveNumber: 1,
        to: regionSpaces[0],
      } as Move;

      await backendAny.applyDecisionMove(regionMoveBackend);

      const backendAfterRegion: GameState = backendEngine.getGameState();
      const backendRegionSnap = snapshotFromBoardAndPlayers(
        'backend-after-region',
        backendAfterRegion.board,
        backendAfterRegion.players
      );

      const backendPendingRegion = backendAny.pendingTerritorySelfElimination === true;
      expect(backendPendingRegion).toBe(true);
      expect(backendAfterRegion.currentPhase).toBe('territory_processing');
      expect(backendAfterRegion.currentPlayer).toBe(movingPlayer);

      if (!snapshotsEqual(backendRegionSnap, sandboxRegionSnap)) {
        // eslint-disable-next-line no-console
        console.error(
          '[TerritoryPendingFlag] mismatch after process_territory_region',
          diffSnapshots(backendRegionSnap, sandboxRegionSnap)
        );
      }
      expect(snapshotsEqual(backendRegionSnap, sandboxRegionSnap)).toBe(true);

      // --- Elimination step: eliminate from the outside stack at outsidePos ---
      // Sandbox: use its internal elimination decision helper.
      const sandboxElimMoves: Move[] =
        sandboxAny.getValidEliminationDecisionMovesForCurrentPlayer() ?? [];
      expect(sandboxElimMoves.length).toBeGreaterThan(0);

      const sandboxElim =
        sandboxElimMoves.find(
          (m: Move) => m.to && m.to.x === outsidePos.x && m.to.y === outsidePos.y
        ) || sandboxElimMoves[0];

      await sandboxEngine.applyCanonicalMove(sandboxElim);

      const sandboxAfterElim: GameState = sandboxEngine.getGameState();
      const sandboxPendingAfterElim = sandboxAny._pendingTerritorySelfElimination === false;
      expect(sandboxPendingAfterElim).toBe(true);

      // Backend: apply an explicit eliminate_rings_from_stack decision for the
      // same outside stack.
      const elimMoveBackend: Move = {
        id: 'eliminate-test',
        type: 'eliminate_rings_from_stack',
        player: movingPlayer,
        to: outsidePos,
        eliminatedRings: [{ player: movingPlayer, count: 1 }],
        eliminationFromStack: {
          position: outsidePos,
          capHeight: 1,
          totalHeight: 1,
        },
        timestamp: new Date(0),
        thinkTime: 0,
        moveNumber: 2,
      } as Move;

      await backendAny.applyDecisionMove(elimMoveBackend);

      const backendAfterElim: GameState = backendEngine.getGameState();
      const backendPendingAfterElim = backendAny.pendingTerritorySelfElimination === false;
      expect(backendPendingAfterElim).toBe(true);

      // Both engines should have left the dedicated territory_processing phase.
      expect(backendAfterElim.currentPhase).not.toBe('territory_processing');
      expect(sandboxAfterElim.currentPhase).not.toBe('territory_processing');

      const backendFinalSnap = snapshotFromBoardAndPlayers(
        'backend-after-elim',
        backendAfterElim.board,
        backendAfterElim.players
      );
      const sandboxFinalSnap = snapshotFromBoardAndPlayers(
        'sandbox-after-elim',
        sandboxAfterElim.board,
        sandboxAfterElim.players
      );

      if (!snapshotsEqual(backendFinalSnap, sandboxFinalSnap)) {
        // eslint-disable-next-line no-console
        console.error(
          '[TerritoryPendingFlag] mismatch after eliminate_rings_from_stack',
          diffSnapshots(backendFinalSnap, sandboxFinalSnap)
        );
      }

      expect(snapshotsEqual(backendFinalSnap, sandboxFinalSnap)).toBe(true);
    });
  }
);
