import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import {
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  Position,
  positionToString,
} from '../../src/shared/types/game';

/**
 * Explicit S-invariant tests for a simple, hand-built sandbox position.
 *
 * Rules/reference:
 * - Compact rules §9 (progress invariant S = markers + collapsed + eliminated)
 * - `ringrift_compact_rules.md` §9 commentary
 *
 * These sandbox-focused tests complement the heavier, diagnostic
 * AI simulations by asserting that:
 *
 * 1. S is computed as M + C + E for a simple sandbox GameState.
 * 2. A canonical "marker → collapsed space + eliminated ring" style
 *    transition strictly increases S in a hand-constructed sandbox state.
 */

describe('ProgressSnapshot (S-invariant) – ClientSandboxEngine (Rules §9)', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice,
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;

        if (anyChoice.type === 'capture_direction') {
          const cd = anyChoice as CaptureDirectionChoice;
          const options = cd.options || [];
          if (options.length === 0) {
            throw new Error('Test SandboxInteractionHandler: no options for capture_direction');
          }

          // Deterministically pick the first option for reproducibility.
          const selected = options[0];
          return {
            choiceId: cd.id,
            playerNumber: cd.playerNumber,
            choiceType: cd.type,
            selectedOption: selected,
          } as PlayerChoiceResponseFor<TChoice>;
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<TChoice>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  it('Rules_9_SInvariant_basic_counts_sandbox', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Start from a clean board and construct a simple configuration:
    // - 1 marker for player 1
    // - 2 collapsed spaces
    // - totalRingsEliminated = 3
    const pos: Position = { x: 0, y: 0 };
    const key = positionToString(pos);

    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();
    state.board.eliminatedRings = {};
    (state as any).totalRingsEliminated = undefined;

    // One marker
    state.board.markers.set(key, { player: 1, position: pos, type: 'regular' });

    // Two collapsed spaces
    state.board.collapsedSpaces.set('1,0', 1);
    state.board.collapsedSpaces.set('2,0', 2);

    // Three eliminated rings via board summary
    state.board.eliminatedRings[1] = 1;
    state.board.eliminatedRings[2] = 2;

    const snap = computeProgressSnapshot(state);
    expect(snap.markers).toBe(1);
    expect(snap.collapsed).toBe(2);
    expect(snap.eliminated).toBe(3);
    expect(snap.S).toBe(1 + 2 + 3);
  });

  it('Rules_9_SInvariant_marker_collapse_increases_S_sandbox', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Construct a tiny board position where:
    // - Player 1 has a marker at (0,0)
    // - No collapsed spaces
    // - No eliminated rings
    const pos: Position = { x: 0, y: 0 };
    const key = positionToString(pos);

    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();
    state.board.eliminatedRings = {};
    state.players.forEach((p) => {
      p.eliminatedRings = 0;
    });
    (state as any).totalRingsEliminated = undefined;

    state.board.markers.set(key, { player: 1, position: pos, type: 'regular' });

    const before = computeProgressSnapshot(state);
    expect(before).toEqual({ markers: 1, collapsed: 0, eliminated: 0, S: 1 });

    // Simulate a canonical progress step where the marker is collapsed
    // into territory and one ring is eliminated for player 1.
    state.board.markers.delete(key);
    state.board.collapsedSpaces.set(key, 1);

    state.board.eliminatedRings[1] = 1;
    const p1 = state.players.find((p) => p.playerNumber === 1)!;
    p1.eliminatedRings = 1;
    (state as any).totalRingsEliminated = 1;

    const after = computeProgressSnapshot(state);
    expect(after.S).toBeGreaterThan(before.S);
    expect(after).toEqual({ markers: 0, collapsed: 1, eliminated: 1, S: 2 });
  });
});
