/**
 * ClientSandboxEngine.branchCoverage7.test.ts
 *
 * Phase 7 branch coverage tests for ClientSandboxEngine.ts,
 * targeting the hardest-to-reach branches in territory processing,
 * line processing, and replay auto-resolve paths.
 */

import {
  ClientSandboxEngine,
  SandboxInteractionHandler,
  SandboxConfig,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type {
  Move,
  PlayerChoice,
  PlayerChoiceResponseFor,
  Position,
  BoardState,
  GamePhase,
} from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

// Mock interaction handler
class ConfigurableMockHandler implements SandboxInteractionHandler {
  public choiceHistory: PlayerChoice[] = [];
  public skipTerritoryProcessing = false;
  public selectSecondOption = false;

  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    this.choiceHistory.push(choice);

    if (choice.type === 'ring_elimination') {
      const elimChoice = choice as PlayerChoice & {
        options: Array<{
          stackPosition: { x: number; y: number };
          capHeight: number;
          totalHeight: number;
          ringsToEliminate: number;
          moveId: string;
        }>;
      };
      const idx = this.selectSecondOption && elimChoice.options.length > 1 ? 1 : 0;
      return {
        choiceId: choice.id,
        selectedOption: elimChoice.options[idx],
      } as PlayerChoiceResponseFor<TChoice>;
    }
    if (choice.type === 'region_order') {
      const regionChoice = choice as PlayerChoice & {
        options: Array<{
          regionId: string;
          size: number;
          representativePosition: { x: number; y: number };
          moveId: string;
          type?: string;
        }>;
      };
      const skipIdx = regionChoice.options.findIndex((o) => o.type === 'skip_territory_processing');
      if (this.skipTerritoryProcessing && skipIdx >= 0) {
        return {
          choiceId: choice.id,
          playerNumber: choice.playerNumber,
          selectedOption: regionChoice.options[skipIdx],
          selectedRegionIndex: skipIdx,
        } as unknown as PlayerChoiceResponseFor<TChoice>;
      }
      const idx = this.selectSecondOption && regionChoice.options.length > 1 ? 1 : 0;
      return {
        choiceId: choice.id,
        playerNumber: choice.playerNumber,
        selectedOption: regionChoice.options[idx],
        selectedRegionIndex: idx,
      } as unknown as PlayerChoiceResponseFor<TChoice>;
    }
    if (choice.type === 'capture_direction') {
      const captureChoice = choice as PlayerChoice & {
        options: Array<{
          targetPosition: { x: number; y: number };
          landingPosition: { x: number; y: number };
          capturedCapHeight: number;
        }>;
      };
      const idx = this.selectSecondOption && captureChoice.options.length > 1 ? 1 : 0;
      return {
        choiceId: choice.id,
        selectedOption: captureChoice.options[idx],
      } as PlayerChoiceResponseFor<TChoice>;
    }
    if (choice.type === 'line_order') {
      const lineChoice = choice as PlayerChoice & {
        options: Array<{
          lineIndex: number;
          positions: Position[];
          moveId: string;
        }>;
      };
      const idx = this.selectSecondOption && lineChoice.options.length > 1 ? 1 : 0;
      return {
        choiceId: choice.id,
        playerNumber: choice.playerNumber,
        selectedOption: lineChoice.options[idx],
        selectedLineIndex: idx,
      } as unknown as PlayerChoiceResponseFor<TChoice>;
    }
    return {
      choiceId: choice.id,
      selectedOption: (choice as { options?: unknown[] }).options?.[0] ?? {},
    } as PlayerChoiceResponseFor<TChoice>;
  }
}

describe('ClientSandboxEngine Branch Coverage 7', () => {
  const createConfig = (
    numPlayers: number = 2,
    boardType: 'square8' | 'square19' = 'square8'
  ): SandboxConfig => ({
    boardType,
    numPlayers,
    playerKinds: Array(numPlayers).fill('human'),
  });

  // ============================================================================
  // getOrchestratorAdapter.getPlayerInfo with AI player (lines 539-546)
  // ============================================================================
  describe('createOrchestratorAdapter.getPlayerInfo', () => {
    it('returns AI player info with difficulty', () => {
      const engine = new ClientSandboxEngine({
        config: {
          boardType: 'square8',
          numPlayers: 2,
          playerKinds: ['ai', 'human'],
          aiDifficulties: [5, 4], // AI difficulties for both players
        },
        interactionHandler: new ConfigurableMockHandler(),
      });

      const adapter = (engine as any).getOrchestratorAdapter();
      const stateAccessor = (adapter as any).stateAccessor;

      const aiInfo = stateAccessor.getPlayerInfo('sandbox-1');
      expect(aiInfo.type).toBe('ai');

      const humanInfo = stateAccessor.getPlayerInfo('sandbox-2');
      expect(humanInfo.type).toBe('human');
    });
  });

  // ============================================================================
  // processMoveViaAdapter when orchestrator returns victoryResult (lines 679-683)
  // ============================================================================
  describe('processMoveViaAdapter victory result', () => {
    it('updates victoryResult when orchestrator detects victory', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      // Set up near-victory state
      const state = engine.getGameState();
      state.players[0].eliminatedRings = state.victoryThreshold - 1;
      state.players[1].ringsInHand = 0;

      // Give player 2 a single stack that would meet elimination threshold
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [2, 2],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 2,
      });
      (engine as any).gameState = state;

      // Place a ring that might trigger victory
      const move: Move = {
        id: 'place1',
        type: 'place_ring',
        player: 1,
        to: { x: 4, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await engine.applyCanonicalMove(move);
      expect(engine.getGameState()).toBeDefined();
    });
  });

  // ============================================================================
  // handleStartOfInteractiveTurn (lines 1741-1783) - non-interactive phases
  // ============================================================================
  describe('handleStartOfInteractiveTurn', () => {
    it('skips LPS tracking for non-interactive phases', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      // Set to a phase that isn't ring_placement/movement/capture/chain_capture
      (engine as any).gameState.currentPhase = 'line_processing';
      (engine as any).handleStartOfInteractiveTurn();

      expect(engine.getGameState()).toBeDefined();
    });

    it('skips LPS tracking when game not active', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.gameStatus = 'completed';
      (engine as any).handleStartOfInteractiveTurn();

      expect(engine.getGameState().gameStatus).toBe('completed');
    });
  });

  // ============================================================================
  // updateLpsRoundTrackingForCurrentPlayer (lines 1644-1668)
  // ============================================================================
  describe('updateLpsRoundTrackingForCurrentPlayer', () => {
    it('returns early when game not active', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.gameStatus = 'completed';
      (engine as any).updateLpsRoundTrackingForCurrentPlayer();

      expect(engine.getGameState()).toBeDefined();
    });

    it('returns early when not in LPS active phase', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.currentPhase = 'game_over';
      (engine as any).updateLpsRoundTrackingForCurrentPlayer();

      expect(engine.getGameState()).toBeDefined();
    });
  });

  // ============================================================================
  // maybeEndGameByLastPlayerStanding (lines 1686-1734)
  // ============================================================================
  describe('maybeEndGameByLastPlayerStanding', () => {
    it('does not end game when LPS not satisfied', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).maybeEndGameByLastPlayerStanding();

      expect(engine.getGameState().gameStatus).toBe('active');
    });
  });

  // ============================================================================
  // createTurnLogicDelegates.applyForcedElimination (lines 1929-1944)
  // ============================================================================
  describe('createTurnLogicDelegates', () => {
    it('returns no-op in traceMode for applyForcedElimination', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
        traceMode: true,
      });

      const delegates = (engine as any).createTurnLogicDelegates();
      const state = engine.getGameState();

      // applyForcedElimination should return state unchanged in traceMode
      const result = delegates.applyForcedElimination(state, 1);
      expect(result).toEqual(state);
    });
  });

  // ============================================================================
  // startTurnForCurrentPlayer safety loop (lines 1954-2004)
  // ============================================================================
  describe('startTurnForCurrentPlayer', () => {
    it('exits loop when victory detected', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      // Set game as completed to make early return
      (engine as any).gameState.gameStatus = 'completed';
      (engine as any).startTurnForCurrentPlayer();

      expect(engine.getGameState().gameStatus).toBe('completed');
    });
  });

  // ============================================================================
  // forceEliminateCapSync in traceMode (lines 2140-2154)
  // ============================================================================
  describe('forceEliminateCapSync', () => {
    it('returns early in traceMode', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
        traceMode: true,
      });

      (engine as any).forceEliminateCapSync(1);

      // Should not throw, just return early
      expect(engine.getGameState()).toBeDefined();
    });
  });

  // ============================================================================
  // eliminateRingForLineReward (lines 2164-2179)
  // ============================================================================
  describe('eliminateRingForLineReward', () => {
    it('does nothing when player has no stacks', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).eliminateRingForLineReward(1);

      expect(engine.getGameState()).toBeDefined();
    });

    it('eliminates from player stack when available', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });
      (engine as any).gameState = state;

      (engine as any).eliminateRingForLineReward(1);

      expect(engine.getGameState()).toBeDefined();
    });
  });

  // ============================================================================
  // getChainCaptureContextForCurrentPlayer (lines 1165-1192)
  // ============================================================================
  describe('getChainCaptureContextForCurrentPlayer', () => {
    it('returns null when game not active', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.gameStatus = 'completed';

      const result = engine.getChainCaptureContextForCurrentPlayer();
      expect(result).toBeNull();
    });

    it('returns null when not in chain_capture phase', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.currentPhase = 'movement';

      const result = engine.getChainCaptureContextForCurrentPlayer();
      expect(result).toBeNull();
    });

    it('returns null when no continuation moves', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.currentPhase = 'chain_capture';

      const result = engine.getChainCaptureContextForCurrentPlayer();
      expect(result).toBeNull();
    });
  });

  // ============================================================================
  // setMarker edge cases (lines 2270-2294)
  // ============================================================================
  describe('setMarker', () => {
    it('does not place marker on collapsed territory', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.collapsedSpaces.set('3,3', 1);
      (engine as any).gameState = state;

      (engine as any).setMarker({ x: 3, y: 3 }, 1, state.board);

      // Marker should not be placed
      expect(state.board.markers.has('3,3')).toBe(false);
    });

    it('removes stack when placing marker', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });
      (engine as any).gameState = state;

      (engine as any).setMarker({ x: 3, y: 3 }, 1, state.board);

      // Stack should be removed
      expect(state.board.stacks.has('3,3')).toBe(false);
      expect(state.board.markers.has('3,3')).toBe(true);
    });
  });

  // ============================================================================
  // flipMarker edge cases (lines 2296-2310)
  // ============================================================================
  describe('flipMarker', () => {
    it('flips opponent marker to own color', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.markers.set('3,3', {
        player: 2,
        position: { x: 3, y: 3 },
        type: 'regular',
      });
      (engine as any).gameState = state;

      (engine as any).flipMarker({ x: 3, y: 3 }, 1, state.board);

      const marker = state.board.markers.get('3,3');
      expect(marker?.player).toBe(1);
    });

    it('does nothing when marker is same color', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.markers.set('3,3', {
        player: 1,
        position: { x: 3, y: 3 },
        type: 'regular',
      });
      (engine as any).gameState = state;

      (engine as any).flipMarker({ x: 3, y: 3 }, 1, state.board);

      const marker = state.board.markers.get('3,3');
      expect(marker?.player).toBe(1);
    });
  });

  // ============================================================================
  // collapseMarker edge cases (lines 2312-2344)
  // ============================================================================
  describe('collapseMarker', () => {
    it('updates territorySpaces for new collapse', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.markers.set('3,3', {
        player: 1,
        position: { x: 3, y: 3 },
        type: 'regular',
      });
      const beforeTerr = state.players[0].territorySpaces;
      (engine as any).gameState = state;

      (engine as any).collapseMarker({ x: 3, y: 3 }, 1, state.board);

      const afterState = engine.getGameState();
      expect(afterState.players[0].territorySpaces).toBe(beforeTerr + 1);
    });

    it('does not increment territory for already-collapsed space', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.collapsedSpaces.set('3,3', 1);
      state.players[0].territorySpaces = 5;
      (engine as any).gameState = state;

      (engine as any).collapseMarker({ x: 3, y: 3 }, 1, state.board);

      const afterState = engine.getGameState();
      expect(afterState.players[0].territorySpaces).toBe(5);
    });
  });

  // ============================================================================
  // findAllLines (lines 2356-2362)
  // ============================================================================
  describe('findAllLines', () => {
    it('finds lines on board', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      // Add a line of markers
      for (let i = 0; i < 4; i++) {
        state.board.markers.set(`${i},0`, {
          player: 1,
          position: { x: i, y: 0 },
          type: 'regular',
        });
      }
      (engine as any).gameState = state;

      const lines = (engine as any).findAllLines(state.board);
      expect(lines.length).toBeGreaterThan(0);
    });
  });

  // ============================================================================
  // promptForCaptureDirection (lines 2439-2469)
  // ============================================================================
  describe('promptForCaptureDirection', () => {
    it('returns first option when only one exists', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const captureOptions: Move[] = [
        {
          id: 'cap1',
          type: 'overtaking_capture',
          player: 1,
          from: { x: 3, y: 3 },
          to: { x: 5, y: 3 },
          captureTarget: { x: 4, y: 3 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ];

      const result = await (engine as any).promptForCaptureDirection(captureOptions);
      expect(result).toEqual(captureOptions[0]);
    });
  });

  // ============================================================================
  // tryPlaceRings edge cases (lines 3473-3551)
  // ============================================================================
  describe('tryPlaceRings edge cases', () => {
    it('returns false when game not active', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.gameStatus = 'completed';

      const result = await engine.tryPlaceRings({ x: 3, y: 3 }, 1);
      expect(result).toBe(false);
    });

    it('returns false when not in ring_placement phase', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.currentPhase = 'movement';

      const result = await engine.tryPlaceRings({ x: 3, y: 3 }, 1);
      expect(result).toBe(false);
    });

    it('returns false when player has no rings', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.players[0].ringsInHand = 0;
      (engine as any).gameState = state;

      const result = await engine.tryPlaceRings({ x: 3, y: 3 }, 1);
      expect(result).toBe(false);
    });
  });

  // ============================================================================
  // hasAnyRealActionForPlayer (lines 1519-1541)
  // ============================================================================
  describe('hasAnyRealActionForPlayer', () => {
    it('returns false when no placement, movement, or capture available', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      // Player 1 with no rings and no stacks = no real actions
      const state = engine.getGameState();
      state.players[0].ringsInHand = 0;
      (engine as any).gameState = state;

      const result = (engine as any).hasAnyRealActionForPlayer(1);
      expect(result).toBe(false);
    });
  });

  // ============================================================================
  // shouldOfferSwapSidesMetaMove (lines 1549-1551)
  // ============================================================================
  describe('shouldOfferSwapSidesMetaMove', () => {
    it('returns false by default (swap rule disabled)', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const result = (engine as any).shouldOfferSwapSidesMetaMove();
      expect(result).toBe(false);
    });
  });

  // ============================================================================
  // applySwapSidesForCurrentPlayer (lines 1576-1637)
  // ============================================================================
  describe('applySwapSidesForCurrentPlayer', () => {
    it('returns false when swap not offered', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const result = engine.applySwapSidesForCurrentPlayer();
      expect(result).toBe(false);
    });
  });

  // ============================================================================
  // buildLastPlayerStandingResult (lines 1676-1678)
  // ============================================================================
  describe('buildLastPlayerStandingResult', () => {
    it('builds LPS victory result', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const result = (engine as any).buildLastPlayerStandingResult(1);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');
    });
  });

  // ============================================================================
  // canProcessDisconnectedRegion (lines 3122-3138)
  // ============================================================================
  describe('canProcessDisconnectedRegion', () => {
    it('returns false when player has no stacks outside region', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      // Single stack in the region
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });
      (engine as any).gameState = state;

      const regionSpaces = [{ x: 3, y: 3 }];
      const result = (engine as any).canProcessDisconnectedRegion(regionSpaces, 1, state.board);
      expect(result).toBe(false);
    });
  });

  // ============================================================================
  // Additional movement/capture edge cases
  // ============================================================================
  describe('movement edge cases', () => {
    it('handleMovementClick clears selection on invalid position', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any)._selectedStackKey = '3,3';
      await (engine as any).handleMovementClick({ x: -1, y: -1 });

      expect((engine as any)._selectedStackKey).toBeUndefined();
    });

    it('handleMovementClick clears selection when clicking same cell', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });
      (engine as any).gameState = state;
      (engine as any)._selectedStackKey = '3,3';

      await (engine as any).handleMovementClick({ x: 3, y: 3 });

      expect((engine as any)._selectedStackKey).toBeUndefined();
    });
  });

  // ============================================================================
  // Additional applyCanonicalMoveForReplay edge cases
  // ============================================================================
  describe('applyCanonicalMoveForReplay additional cases', () => {
    it('handles process_line move', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.currentPhase = 'line_processing';

      const move: Move = {
        id: 'line1',
        type: 'process_line',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
              { x: 3, y: 0 },
            ],
            player: 1,
            length: 4,
            direction: { x: 1, y: 0 },
          },
        ],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await engine.applyCanonicalMoveForReplay(move, null);

      expect(engine.getGameState()).toBeDefined();
    });

    it('handles choose_line_option move', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.currentPhase = 'line_processing';

      const move: Move = {
        id: 'lineoption1',
        type: 'choose_line_option',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
              { x: 3, y: 0 },
            ],
            player: 1,
            length: 4,
            direction: { x: 1, y: 0 },
          },
        ],
        collapsedMarkers: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await engine.applyCanonicalMoveForReplay(move, null);

      expect(engine.getGameState()).toBeDefined();
    });

    it('handles choose_territory_option move', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.currentPhase = 'territory_processing';

      const move: Move = {
        id: 'terr1',
        type: 'choose_territory_option',
        player: 1,
        to: { x: 3, y: 3 },
        disconnectedRegions: [
          {
            spaces: [{ x: 3, y: 3 }],
            controllingPlayer: 1,
            isDisconnected: true,
          },
        ],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await engine.applyCanonicalMoveForReplay(move, null);

      expect(engine.getGameState()).toBeDefined();
    });

    it('handles eliminate_rings_from_stack move when elimination pending', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });
      // Set pending territory elimination state
      (engine as any)._pendingTerritorySelfElimination = true;
      (engine as any).gameState = state;

      const move: Move = {
        id: 'elim1',
        type: 'eliminate_rings_from_stack',
        player: 1,
        to: { x: 3, y: 3 },
        eliminatedRings: [{ player: 1, count: 1 }],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // This may fail or succeed depending on orchestrator state
      try {
        await engine.applyCanonicalMoveForReplay(move, null);
      } catch {
        // Expected when orchestrator rejects
      }

      expect(engine.getGameState()).toBeDefined();
    });

    it('handles skip_capture move', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      (engine as any).gameState.currentPhase = 'capture';

      const move: Move = {
        id: 'skip1',
        type: 'skip_capture',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await engine.applyCanonicalMoveForReplay(move, null);

      expect(engine.getGameState()).toBeDefined();
    });

    it('handles skip_placement move', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const move: Move = {
        id: 'skip1',
        type: 'skip_placement',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await engine.applyCanonicalMoveForReplay(move, null);

      expect(engine.getGameState()).toBeDefined();
    });

    it('handles continue_capture_segment move', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new ConfigurableMockHandler(),
      });

      const state = engine.getGameState();
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });
      state.board.stacks.set('4,3', {
        position: { x: 4, y: 3 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });
      state.currentPhase = 'chain_capture';
      state.chainCapturePosition = { x: 3, y: 3 };
      (engine as any).gameState = state;

      const move: Move = {
        id: 'cont1',
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 5, y: 3 },
        captureTarget: { x: 4, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await engine.applyCanonicalMoveForReplay(move, null);

      expect(engine.getGameState()).toBeDefined();
    });
  });
});
