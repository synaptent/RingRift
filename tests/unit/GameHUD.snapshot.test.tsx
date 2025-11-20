import React from 'react';
import { renderToString } from 'react-dom/server';
import { GameHUD } from '../../src/client/components/GameHUD';
import { BOARD_CONFIGS, GameState, Player } from '../../src/shared/types/game';

function createTestGameState(): { gameState: GameState; currentPlayer: Player } {
  const boardConfig = BOARD_CONFIGS.square8;

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Alice',
      type: 'human',
      playerNumber: 1,
      rating: 1500,
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 10,
      eliminatedRings: 3,
      territorySpaces: 8,
    },
    {
      id: 'p2',
      username: 'Bob',
      type: 'ai',
      playerNumber: 2,
      rating: 1400,
      isReady: true,
      timeRemaining: 4 * 60 * 1000,
      ringsInHand: 12,
      eliminatedRings: 1,
      territorySpaces: 5,
    },
  ];

  const now = new Date();

  const gameState: GameState = {
    id: 'game-1',
    boardType: 'square8',
    board: {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: boardConfig.size,
      type: 'square8',
    },
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: {
      type: 'rapid',
      initialTime: 600,
      increment: 0,
    },
    spectators: [],
    gameStatus: 'active',
    createdAt: now,
    lastMoveAt: now,
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: boardConfig.ringsPerPlayer * players.length,
    totalRingsEliminated: players.reduce((sum, p) => sum + p.eliminatedRings, 0),
    victoryThreshold:
      Math.floor((boardConfig.ringsPerPlayer * players.length) / 2) + 1,
    territoryVictoryThreshold: Math.floor(boardConfig.totalSpaces / 2) + 1,
  };

  return { gameState, currentPlayer: players[0] };
}

describe('GameHUD snapshot', () => {
  it('renders a compact HUD with per-player summary without crashing', () => {
    const { gameState, currentPlayer } = createTestGameState();

    const html = renderToString(
      <GameHUD
        gameState={gameState}
        currentPlayer={currentPlayer}
        instruction="Select a stack to move."
        connectionStatus="connected"
        isSpectator={false}
        lastHeartbeatAt={Date.now()}
      />
    );

    expect(html).toMatchSnapshot();
  });
});
