import { ClientSandboxEngine, SandboxConfig, SandboxInteractionHandler } from '../../src/client/sandbox/ClientSandboxEngine';
import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import {
  BoardType,
  GameState,
  Position,
  Move,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString
} from '../../src/shared/types/game';
import {
  createTestGameState,
  createTestBoard,
  createTestPlayer,
  addStack,
  addMarker
} from '../utils/fixtures';

/**
 * Parity test: sandbox movement landings vs backend RuleEngine.getValidMoves.
 *
 * For a small handcrafted position, we compare:
 * - sandbox: ClientSandboxEngine.getValidLandingPositionsForCurrentPlayer(from), and
 * - backend: RuleEngine.getValidMoves(gameState) filtered to movement moves
 *   originating from the same source stack.
 *
 * This helps ensure sandbox move gating stays aligned with backend
 * semantics (path rules, marker rules, stack merging landings).
 */

describe('ClientSandboxEngine movement parity with RuleEngine', () => {
  const boardType: BoardType = 'square8';

  function createSandboxEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human']
    };

    const handler: SandboxInteractionHandler = {
      // Movement parity tests dont depend on PlayerChoices; provide a
      // trivial handler that always picks the first option when invoked.
      async requestChoice<TChoice extends any>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options ? (anyChoice as any).options[0] : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption
        } as PlayerChoiceResponseFor<any>;
      }
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  test('sandbox landing positions match backend move_stack targets for a simple scenario', () => {
    const engine = createSandboxEngine();
    const engineAny = engine as any;
    const sandboxState: GameState = engineAny.gameState as GameState;

    // Configure current player / phase to match a normal movement turn.
    sandboxState.currentPlayer = 1;
    sandboxState.currentPhase = 'movement';

    const board = sandboxState.board;

    // Clear default board content just in case.
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    // Source stack for player 1 at (3,3) with height 2.
    const from: Position = { x: 3, y: 3 };
    addStack(board, from, 1, 2);

    // Blocker stack (player 2) directly north of source to exercise
    // path-blocking. This should prevent moving through (3,2).
    addStack(board, { x: 3, y: 2 }, 2, 1);

    // Same-colour marker one step east; landing there is allowed.
    addMarker(board, { x: 4, y: 3 }, 1);

    // Opponent marker one step west; landing there must be disallowed.
    addMarker(board, { x: 2, y: 3 }, 2);

    // Simple empty landing two steps east; also allowed.
    // Board size is 8x8 so everything is on-board.

    // --- Sandbox side: enumerate simple (non-capturing) landing positions
    // from the source stack using the same helper the AI uses. This mirrors
    // the backend movement phase semantics (captures are handled in a
    // separate phase in the backend engine).
    const simpleLandings: Array<{ fromKey: string; to: Position }> = (engine as any)
      .enumerateSimpleMovementLandings(1);
    const sandboxLandingKeys = simpleLandings
      .filter(m => m.fromKey === positionToString(from))
      .map(m => positionToString(m.to))
      .sort();

    // --- Backend side: mirror board into a GameState and ask RuleEngine ---
    const backendBoard = createTestBoard(boardType);
    // Copy stacks/markers/collapsedSpaces into backendBoard to mirror sandbox.
    for (const [key, stack] of board.stacks.entries()) {
      backendBoard.stacks.set(key, { ...stack });
    }
    for (const [key, marker] of board.markers.entries()) {
      backendBoard.markers.set(key, { ...marker });
    }
    for (const [key, owner] of board.collapsedSpaces.entries()) {
      backendBoard.collapsedSpaces.set(key, owner);
    }

    const backendGameState = createTestGameState({
      boardType,
      board: backendBoard,
      players: [
        createTestPlayer(1, { type: 'human', ringsInHand: 0 }),
        createTestPlayer(2, { type: 'human', ringsInHand: 0 })
      ],
      currentPlayer: 1,
      currentPhase: 'movement'
    });

    const boardManager = new BoardManager(boardType);
    const ruleEngine = new RuleEngine(boardManager, boardType as any);

    const backendMoves: Move[] = ruleEngine.getValidMoves(backendGameState);
    const backendMovementTargets = backendMoves
      .filter(m => m.type === 'move_stack' && m.from && positionToString(m.from) === positionToString(from))
      .map(m => positionToString(m.to))
      .sort();

    const backendTargetSet = new Set(backendMovementTargets);

    // Parity guarantee we care about: every sandbox-legal landing is also
    // accepted by the backend RuleEngine as a valid move_stack target from
    // the same source stack. The backend may allow additional moves (due to
    // its more global search), but the sandbox must never surface a move
    // that the backend would reject.
    for (const key of sandboxLandingKeys) {
      expect(backendTargetSet.has(key)).toBe(true);
    }
  });
});
