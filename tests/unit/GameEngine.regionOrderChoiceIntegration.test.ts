import { EventEmitter } from 'events';
import { GameEngine } from '../../src/server/game/GameEngine';
import { PlayerInteractionManager } from '../../src/server/game/PlayerInteractionManager';
import { WebSocketInteractionHandler } from '../../src/server/game/WebSocketInteractionHandler';
import {
  BoardType,
  GameState,
  Player,
  Position,
  Territory,
  TimeControl,
  RegionOrderChoice,
  PlayerChoiceResponse
} from '../../src/shared/types/game';

// Minimal Socket.IO Server stub for testing end-to-end choice plumbing
class FakeSocketIOServer extends EventEmitter {
  public toCalls: Array<{ target: string; event: string; payload: any }> = [];

  to(target: string) {
    return {
      emit: (event: string, payload: any) => {
        this.toCalls.push({ target, event, payload });
        this.emit(event, payload);
      }
    };
  }
}

const boardType: BoardType = 'square8';
const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

const players: Player[] = [
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

/**
 * RegionOrderChoice integration test:
 *   GameEngine.processDisconnectedRegions → PlayerInteractionManager →
 *   WebSocketInteractionHandler → fake Socket.IO client →
 *   WebSocketInteractionHandler.handleChoiceResponse →
 *   GameEngine.processOneDisconnectedRegion.
 *
 * We stub BoardManager.findDisconnectedRegions to return two synthetic
 * Territory regions and spy on processOneDisconnectedRegion to
 * ensure the region selected by the client is the one processed
 * first.
 */

describe('GameEngine + WebSocketInteractionHandler region order choice integration', () => {
  it('emits RegionOrderChoice and processes the region selected by the client first', async () => {
    const io = new FakeSocketIOServer();

    const getTargetForPlayer = jest.fn().mockReturnValue('socket-1');
    const handler = new WebSocketInteractionHandler(
      io as any,
      'region-order-game',
      getTargetForPlayer,
      30_000
    );
    const interactionManager = new PlayerInteractionManager(handler);

    const engine = new GameEngine(
      'region-order-game',
      boardType,
      players,
      timeControl,
      false,
      interactionManager
    );

    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as GameState;

    gameState.currentPlayer = 1;

    // Two synthetic regions with distinct representative positions so we
    // can distinguish them easily.
    const regionA: Territory = {
      spaces: [
        { x: 1, y: 1 },
        { x: 1, y: 2 }
      ],
      controllingPlayer: 0,
      isDisconnected: true
    };

    const regionB: Territory = {
      spaces: [
        { x: 5, y: 5 },
        { x: 5, y: 6 }
      ],
      controllingPlayer: 0,
      isDisconnected: true
    };

    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementationOnce(() => [regionA, regionB])
      .mockImplementation(() => []);

    // For this integration test we are interested in the ordering
    // behaviour rather than the self-elimination prerequisite, so
    // stub canProcessDisconnectedRegion to always allow processing.
    jest.spyOn(engineAny, 'canProcessDisconnectedRegion').mockReturnValue(true);

    const processRegionSpy = jest
      .spyOn(engineAny, 'processOneDisconnectedRegion')
      .mockResolvedValue(undefined);

    // Kick off the disconnected-region processing loop.
    const processPromise: Promise<void> = engineAny.processDisconnectedRegions();

    expect(getTargetForPlayer).toHaveBeenCalledWith(1);
    expect(io.toCalls).toHaveLength(1);

    const call = io.toCalls[0];
    expect(call.event).toBe('player_choice_required');

    const choice = call.payload as RegionOrderChoice;
    expect(choice.type).toBe('region_order');
    expect(choice.playerNumber).toBe(1);
    expect(choice.options.length).toBe(2);

    // The RegionOrderChoice options are ordered according to the
    // disconnectedRegions array. We want to select the SECOND region
    // (regionB) and ensure it is processed first.
    const selectedOption = choice.options[1];

    const response: PlayerChoiceResponse<(typeof choice.options)[number]> = {
      choiceId: choice.id,
      playerNumber: choice.playerNumber,
      selectedOption
    };

    handler.handleChoiceResponse(response as any);

    await processPromise;

    // Ensure BoardManager.findDisconnectedRegions was consulted and the
    // chosen region (regionB) was passed first to processOneDisconnectedRegion.
    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();
    expect(processRegionSpy).toHaveBeenCalled();

    const firstCallArgs = processRegionSpy.mock.calls[0] as [Territory, number];
    const firstRegion = firstCallArgs[0];

    const toKeySet = (spaces: Position[]) => new Set(spaces.map(p => `${p.x},${p.y}`));
    const regionBKeys = toKeySet(regionB.spaces);
    const firstRegionKeys = toKeySet(firstRegion.spaces);

    expect(firstRegionKeys).toEqual(regionBKeys);
  });
});
