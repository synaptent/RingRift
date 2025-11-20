import React from 'react';
import { renderToString } from 'react-dom/server';
import { GameEventLog } from '../../src/client/components/GameEventLog';
import {
  GameHistoryEntry,
  GamePhase,
  GameStatus,
  Move,
  Position,
  ProgressSnapshot,
} from '../../src/shared/types/game';

function pos(x: number, y: number, z?: number): Position {
  return z === undefined ? { x, y } : { x, y, z };
}

function createHistoryEntry(overrides: Partial<GameHistoryEntry> & { action: Move }): GameHistoryEntry {
  const progress: ProgressSnapshot = { markers: 0, collapsed: 0, eliminated: 0, S: 0 };
  const now = new Date();

  return {
    moveNumber: overrides.moveNumber ?? overrides.action.moveNumber ?? 1,
    action: overrides.action,
    actor: overrides.action.player,
    phaseBefore: (overrides.phaseBefore as GamePhase) ?? 'movement',
    phaseAfter: (overrides.phaseAfter as GamePhase) ?? 'movement',
    statusBefore: (overrides.statusBefore as GameStatus) ?? 'active',
    statusAfter: (overrides.statusAfter as GameStatus) ?? 'active',
    progressBefore: overrides.progressBefore ?? progress,
    progressAfter: overrides.progressAfter ?? progress,
    stateHashBefore: overrides.stateHashBefore ?? 'before-hash',
    stateHashAfter: overrides.stateHashAfter ?? 'after-hash',
    boardBeforeSummary: overrides.boardBeforeSummary,
    boardAfterSummary: overrides.boardAfterSummary,
  };
}

describe('GameEventLog snapshot', () => {
  it('renders recent moves and system events without crashing', () => {
    const moves: Move[] = [
      {
        id: 'm1',
        type: 'place_ring',
        player: 1,
        to: pos(0, 0),
        placementCount: 2,
        timestamp: new Date(),
        thinkTime: 100,
        moveNumber: 1,
      },
      {
        id: 'm2',
        type: 'overtaking_capture',
        player: 2,
        from: pos(3, 3),
        to: pos(3, 5),
        captureTarget: pos(3, 4),
        overtakenRings: [1, 1],
        timestamp: new Date(),
        thinkTime: 250,
        moveNumber: 2,
      },
      {
        id: 'm3',
        type: 'process_territory_region',
        player: 1,
        to: pos(0, 0),
        eliminatedRings: [
          { player: 2, count: 2 },
        ],
        timestamp: new Date(),
        thinkTime: 50,
        moveNumber: 3,
      },
    ];

    const history: GameHistoryEntry[] = [
      createHistoryEntry({ action: moves[0] }),
      createHistoryEntry({ action: moves[1] }),
      createHistoryEntry({ action: moves[2] }),
    ];

    const systemEvents = [
      'Phase: ring_placement',
      'Current player: P1',
      'Phase changed: movement → capture',
      'Connection restored',
    ];

    const html = renderToString(
      <GameEventLog history={history} systemEvents={systemEvents} victoryState={null} />
    );

    expect(html).toMatchSnapshot();
  });
});
