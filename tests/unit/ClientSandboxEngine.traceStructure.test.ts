import { ClientSandboxEngine, SandboxConfig, SandboxInteractionHandler } from '../../src/client/sandbox/ClientSandboxEngine';
import { BoardType, GameTrace, GameHistoryEntry, Move, PlayerChoice, PlayerChoiceResponseFor } from '../../src/shared/types/game';

/**
 * Minimal deterministic SandboxInteractionHandler for tests. All choices
 * simply pick the first option so flows are reproducible.
 */
class TestSandboxInteractionHandler implements SandboxInteractionHandler {
  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    const anyChoice = choice as any;
    const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
    return {
      choiceId: anyChoice.id,
      playerNumber: anyChoice.playerNumber,
      choiceType: anyChoice.type,
      selectedOption
    } as PlayerChoiceResponseFor<TChoice>;
  }
}

function createTwoPlayerSandbox(boardType: BoardType = 'square8'): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType,
    numPlayers: 2,
    playerKinds: ['ai', 'ai']
  };

  const handler = new TestSandboxInteractionHandler();
  return new ClientSandboxEngine({ config, interactionHandler: handler });
}

function extractTrace(engine: ClientSandboxEngine): GameTrace {
  const initialState = engine.getGameState();
  return {
    initialState,
    entries: initialState.history
  };
}

describe('ClientSandboxEngine trace structure', () => {
  it('emits contiguous moveNumbers starting from 1 for simple AI turns', async () => {
    const engine = createTwoPlayerSandbox('square8');

    // Drive a few AI turns. We deliberately keep this small so the test
    // stays cheap while still exercising placement → movement → capture
    // transitions in typical positions.
    for (let i = 0; i < 10; i++) {
      const state = engine.getGameState();
      if (state.gameStatus !== 'active') break;
      const current = state.players.find(p => p.playerNumber === state.currentPlayer);
      if (!current || current.type !== 'ai') break;
      await engine.maybeRunAITurn();
    }

    const trace = extractTrace(engine);
    const entries = trace.entries;

    expect(entries.length).toBeGreaterThan(0);

    // moveNumber should be 1..N without gaps.
    entries.forEach((entry, index) => {
      expect(entry.moveNumber).toBe(index + 1);
      expect(entry.actor).toBe(entry.action.player);
    });

    // Sanity check: progress snapshots should always have S >= 0 and
    // be defined for all entries.
    for (const entry of entries) {
      expect(entry.progressBefore).toBeDefined();
      expect(entry.progressAfter).toBeDefined();
      expect(entry.progressBefore.S).toBeGreaterThanOrEqual(0);
      expect(entry.progressAfter.S).toBeGreaterThanOrEqual(0);
    }
  });

  it('includes explicit skip_placement steps when AI has no legal placements', async () => {
    const engine = createTwoPlayerSandbox('square8');

    // Artificially construct a situation where player 1 has rings in hand
    // but no legal placements that satisfy no-dead-placement, by clearing
    // the board of stacks and collapsing all spaces. This is extreme but
    // guarantees enumerateLegalRingPlacements returns [] and forces the
    // skip_placement path.
    const stateBefore = engine.getGameState();
    const board = stateBefore.board;
    const collapsed = new Map(board.collapsedSpaces);

    // Collapse every coordinate on the 8x8 board.
    for (let x = 0; x < board.size; x++) {
      for (let y = 0; y < board.size; y++) {
        const key = `${x},${y}`;
        collapsed.set(key, 1);
      }
    }

    const mutatedBoard: any = {
      ...board,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: collapsed
    };

    (engine as any).gameState = {
      ...stateBefore,
      board: mutatedBoard,
      currentPhase: 'ring_placement',
      currentPlayer: 1
    };

    await engine.maybeRunAITurn();

    const trace = extractTrace(engine);
    const entries = trace.entries;

    expect(entries.length).toBeGreaterThan(0);

    const skipEntry = entries.find(e => e.action.type === 'skip_placement');
    expect(skipEntry).toBeDefined();
    expect(skipEntry!.action.player).toBe(1);
    expect(skipEntry!.phaseBefore).toBe('ring_placement');
    expect(skipEntry!.phaseAfter).toBe('movement');
  });
});
