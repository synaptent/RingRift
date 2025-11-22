jest.mock('../../src/shared/utils/envFlags', () => {
  const actual = jest.requireActual('../../src/shared/utils/envFlags');
  return {
    ...actual,
    isSandboxAiParityModeEnabled: jest.fn(),
  };
});

jest.mock('../../src/shared/engine/localAIMoveSelection', () => {
  const actual = jest.requireActual('../../src/shared/engine/localAIMoveSelection');
  return {
    ...actual,
    chooseLocalMoveFromCandidates: jest.fn(),
  };
});

import { GameState, Position, RingStack } from '../../src/shared/types/game';
import { createTestGameState, pos } from '../utils/fixtures';
import {
  buildSandboxMovementCandidates,
  selectSandboxMovementMove,
} from '../../src/client/sandbox/sandboxAI';
import type { SandboxAIHooks } from '../../src/client/sandbox/sandboxAI';
import { isSandboxAiParityModeEnabled } from '../../src/shared/utils/envFlags';
import { chooseLocalMoveFromCandidates } from '../../src/shared/engine/localAIMoveSelection';

describe('Sandbox AI movement parity mode wiring', () => {
  const parityModeMock = isSandboxAiParityModeEnabled as jest.MockedFunction<
    typeof isSandboxAiParityModeEnabled
  >;

  const chooseLocalMoveFromCandidatesMock = chooseLocalMoveFromCandidates as jest.MockedFunction<
    typeof chooseLocalMoveFromCandidates
  >;

  it('builds movement candidates deterministically and selects via shared policy in both parity modes', () => {
    const baseState: GameState = createTestGameState({
      boardType: 'square8',
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    let currentState: GameState = {
      ...baseState,
      players: baseState.players.map((p) =>
        p.playerNumber === 1 ? { ...p, type: 'ai' } : { ...p, type: 'human' }
      ),
      gameStatus: 'active',
      history: [],
    };

    const stackPosition: Position = pos(3, 3);
    const stack: RingStack = {
      position: stackPosition,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    };

    const captureTarget: Position = pos(4, 3);
    const captureLanding: Position = pos(5, 3);
    const simpleLanding: Position = pos(6, 3);

    const hooks: SandboxAIHooks = {
      getPlayerStacks: () => [stack],
      hasAnyLegalMoveOrCaptureFrom: () => false,
      enumerateLegalRingPlacements: () => [],
      tryPlaceRings: () => false,
      enumerateCaptureSegmentsFrom: () => [
        { from: stackPosition, target: captureTarget, landing: captureLanding },
      ],
      enumerateSimpleMovementLandings: () => [
        { fromKey: `${stackPosition.x},${stackPosition.y}`, to: simpleLanding },
      ],
      maybeProcessForcedEliminationForCurrentPlayer: () => false,
      handleMovementClick: async () => {
        // no-op for this helper wiring test
      },
      appendHistoryEntry: () => {
        // history is driven by ClientSandboxEngine in production; not needed here
      },
      getGameState: () => currentState,
      setGameState: (state: GameState) => {
        currentState = state;
      },
      setLastAIMove: () => {
        // tracked internally by sandboxAI; not needed here
      },
      setSelectedStackKey: () => {
        // selection not relevant for this test
      },
      getMustMoveFromStackKey: () => undefined,
      applyCanonicalMove: async () => {
        // no-op: this test focuses only on candidate building + selection wiring
      },
    };

    const rng = () => 0.5;

    const { candidates, debug } = buildSandboxMovementCandidates(currentState, hooks, rng);

    expect(debug.captureCount).toBe(1);
    expect(debug.simpleMoveCount).toBe(1);
    expect(candidates).toHaveLength(2);

    // === Parity mode enabled: selection delegates to shared policy ===
    parityModeMock.mockReturnValue(true);
    chooseLocalMoveFromCandidatesMock.mockClear();

    selectSandboxMovementMove(currentState, [...candidates], rng, true);

    expect(chooseLocalMoveFromCandidatesMock).toHaveBeenCalledTimes(1);
    const [playerOn, stateOn, candidatesOn, rngOn] =
      chooseLocalMoveFromCandidatesMock.mock.calls[0];

    expect(playerOn).toBe(1);
    expect(stateOn).toBe(currentState);
    expect(candidatesOn).toEqual(candidates);
    expect(rngOn).toBe(rng);

    // === Parity mode disabled: currently identical behaviour but separate branch ===
    parityModeMock.mockReturnValue(false);
    chooseLocalMoveFromCandidatesMock.mockClear();

    selectSandboxMovementMove(currentState, [...candidates], rng, false);

    expect(chooseLocalMoveFromCandidatesMock).toHaveBeenCalledTimes(1);
    const [playerOff, stateOff, candidatesOff, rngOff] =
      chooseLocalMoveFromCandidatesMock.mock.calls[0];

    expect(playerOff).toBe(playerOn);
    expect(stateOff).toBe(stateOn);
    expect(candidatesOff).toEqual(candidatesOn);
    expect(rngOff).toBe(rngOn);
  });
});
