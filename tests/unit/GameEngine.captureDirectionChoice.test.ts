import { GameEngine } from '../../src/server/game/GameEngine';
import { PlayerInteractionManager } from '../../src/server/game/PlayerInteractionManager';
import {
  BoardType,
  Move,
  Player,
  Position,
  TimeControl,
  PlayerChoice,
  PlayerChoiceResponse
} from '../../src/shared/types/game';

/**
 * Tests for wiring CaptureDirectionChoice into GameEngine via
 * chainCaptureState.availableMoves and PlayerInteractionManager.
 *
 * These tests focus on ensuring that when multiple follow-up capture
 * options exist in chainCaptureState.availableMoves, GameEngine
 * constructs a CaptureDirectionChoice and returns the Move that
 * matches the player's selected option.
 */

describe('GameEngine capture direction choice wiring', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  function createEngineWithInteraction(
    handler: { requestChoice: (choice: PlayerChoice) => Promise<PlayerChoiceResponse<unknown>> }
  ): GameEngine {
    const interactionManager = new PlayerInteractionManager(handler as any);
    return new GameEngine('test-game-capture-dir', boardType, basePlayers, timeControl, false, interactionManager);
  }

  test('uses CaptureDirectionChoice to select among multiple chain capture options', async () => {
    const mockHandler = {
      requestChoice: jest.fn(async (choice: PlayerChoice): Promise<PlayerChoiceResponse<any>> => {
        expect(choice.type).toBe('capture_direction');
        expect(choice.options.length).toBe(2);

        // Choose the second option to verify mapping back to Move
        const selectedOption = choice.options[1];

        return {
          choiceId: choice.id,
          playerNumber: choice.playerNumber,
          selectedOption
        };
      })
    };

    const engine = createEngineWithInteraction(mockHandler);

    // Simulate a chain in progress for player 1 from position (5,5)
    const currentPos: Position = { x: 5, y: 5 };
    const targetA: Position = { x: 6, y: 6 };
    const landingA: Position = { x: 7, y: 7 };
    const targetB: Position = { x: 4, y: 6 };
    const landingB: Position = { x: 3, y: 7 };

    const optionA: Move = {
      id: 'opt-a',
      type: 'overtaking_capture',
      player: 1,
      from: currentPos,
      captureTarget: targetA,
      to: landingA,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const optionB: Move = {
      id: 'opt-b',
      type: 'overtaking_capture',
      player: 1,
      from: currentPos,
      captureTarget: targetB,
      to: landingB,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    // Force the internal chain capture state with two available options
    (engine as any).chainCaptureState = {
      playerNumber: 1,
      startPosition: { x: 3, y: 3 },
      currentPosition: currentPos,
      segments: [],
      availableMoves: [optionA, optionB],
      visitedPositions: new Set<string>(['3,3', '5,5'])
    };

    // Call the internal helper via `any` to avoid changing visibility
    const chosen: Move | undefined = await (engine as any).chooseCaptureDirectionFromState();

    expect(mockHandler.requestChoice).toHaveBeenCalledTimes(1);
    // Because the mock handler selected the second option, we expect
    // the helper to return optionB.
    expect(chosen).toBe(optionB);
  });
});
