import { PythonRulesClient } from '../../src/server/services/PythonRulesClient';
import { GameEngine } from '../../src/server/game/GameEngine';
import type { GameState, Move, Player, BoardType, TimeControl } from '../../src/shared/types/game';
import { BOARD_CONFIGS } from '../../src/shared/engine';

/**
 * Optional live-integration parity test for the 2-player pie rule (swap_sides).
 *
 * This test:
 *   - Constructs a minimal 2p square8 GameState with rulesOptions.swapRuleEnabled=true.
 *   - Applies Player 1's first placement via the TS GameEngine.
 *   - Synthesises the start of Player 2's interactive turn and constructs a
 *     canonical swap_sides Move matching TS semantics.
 *   - Sends that move to the Python /rules/evaluate_move endpoint via
 *     PythonRulesClient.
 *   - Independently applies the same move via TS GameEngine.applySwapSidesMove
 *     (via makeMove) and compares the resulting GameState hash/metadata
 *     against Python's nextState to assert TS↔Python parity for swap_sides.
 *
 * Like the other PythonRulesClient live tests, this suite is SKIPPED by
 * default unless:
 *
 *   - RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION is "1" or "true", and
 *   - AI_SERVICE_URL is set to the live ai-service base URL.
 */

const ENABLED =
  (process.env.RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION === '1' ||
    process.env.RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION === 'true' ||
    process.env.RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION === 'TRUE') &&
  typeof process.env.AI_SERVICE_URL === 'string' &&
  process.env.AI_SERVICE_URL.length > 0;

function createTwoPlayerStateWithSwapRule(boardType: BoardType = 'square8'): GameState {
  const now = new Date();
  const boardConfig = BOARD_CONFIGS[boardType];

  const players: Player[] = [
    {
      id: 'p1',
      username: 'P1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      aiDifficulty: undefined,
      ringsInHand: boardConfig.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'P2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      aiDifficulty: undefined,
      ringsInHand: boardConfig.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  const timeControl: TimeControl = {
    type: 'rapid',
    initialTime: 600,
    increment: 0,
  };

  const engine = new GameEngine('swap-rule-live', boardType, players, timeControl, false);

  // Enable the pie rule for this game.
  const stateWithRule: GameState = {
    ...engine.getGameState(),
    rulesOptions: { swapRuleEnabled: true },
  };
  (engine as any).gameState = stateWithRule;

  return engine.getGameState();
}

function hashLightState(state: GameState): string {
  const playersMeta = state.players
    .map((p) => `${p.playerNumber}:${p.id}:${p.username}:${p.type}:${p.aiDifficulty ?? ''}:${p.ringsInHand}:${p.eliminatedRings}:${p.territorySpaces}`)
    .sort()
    .join('|');

  return [
    state.boardType,
    state.currentPlayer,
    state.currentPhase,
    state.gameStatus,
    playersMeta,
  ].join('#');
}

(ENABLED ? describe : describe.skip)(
  'PythonRulesClient pie rule (swap_sides) parity against TS backend',
  () => {
    const baseUrl = process.env.AI_SERVICE_URL as string;

    it('applies swap_sides via /rules/evaluate_move with TS↔Python parity', async () => {
      const client = new PythonRulesClient(baseUrl);

      // 1. Create a 2p state with swap rule enabled and let TS apply P1's first move.
      const initialState = createTwoPlayerStateWithSwapRule('square8');
      const boardConfig = BOARD_CONFIGS[initialState.boardType];

      const timeControl: TimeControl = {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      };
      const tsEngine = new GameEngine(
        initialState.id,
        initialState.boardType,
        initialState.players,
        timeControl,
        false
      );
      (tsEngine as any).gameState = initialState;

      const p1Moves = tsEngine.getValidMoves(1);
      const firstPlacement = p1Moves.find((m) => m.type === 'place_ring');
      expect(firstPlacement).toBeDefined();

      const afterP1Result = await tsEngine.makeMove(firstPlacement!);
      expect(afterP1Result.success).toBe(true);
      const afterP1State = afterP1Result.gameState!;

      // 2. Synthesize the start of P2's interactive turn for gating parity.
      const forP2Turn: GameState = {
        ...afterP1State,
        currentPlayer: 2,
      };

      // 3. Construct a canonical swap_sides Move matching TS semantics.
      const swapMoveNumber = forP2Turn.moveHistory.length + 1;
      const swapMove: Move = {
        id: `swap_sides-${swapMoveNumber}`,
        type: 'swap_sides',
        player: 2,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: swapMoveNumber,
      } as Move;

      // 4. Ask Python to evaluate and apply swap_sides.
      const pyResult = await client.evaluateMove(forP2Turn, swapMove);
      expect(pyResult.valid).toBe(true);
      expect(pyResult.nextState).toBeDefined();

      const pyNext = pyResult.nextState!;

      // 5. Apply swap_sides via TS backend and compare light hashes + key meta.
      (tsEngine as any).gameState = forP2Turn;
      const tsSwapResult = await tsEngine.makeMove({
        type: 'swap_sides',
        player: 2,
      } as any);
      expect(tsSwapResult.success).toBe(true);
      const tsNext = tsSwapResult.gameState!;

      expect(hashLightState(pyNext)).toBe(hashLightState(tsNext));
      expect(pyNext.currentPlayer).toBe(tsNext.currentPlayer);
      expect(pyNext.currentPhase).toBe(tsNext.currentPhase);
      expect(pyNext.gameStatus).toBe(tsNext.gameStatus);
    });
  }
);

