/**
 * Shared test utilities for SandboxGameHost tests.
 *
 * This file contains helper functions used across the split test files.
 * Each test file has its own jest.mock() declarations due to Jest's hoisting requirements.
 */

import { screen } from '@testing-library/react';
import type { BoardState, GameState, Player, Position } from '../../../src/shared/types/game';
import type { LocalConfig } from '../../../src/client/contexts/SandboxContext';

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

export function createEmptySquareBoard(size = 8): BoardState {
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size,
    type: 'square8',
  };
}

export function createPlayers(overrides: Partial<Player>[] = []): Player[] {
  const base: Player[] = [
    {
      id: 'p1',
      username: 'P1',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'P2',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
      aiDifficulty: 5,
      aiProfile: { difficulty: 5, aiType: 'heuristic' },
    },
  ];

  return base.map((p, idx) => ({ ...p, ...(overrides[idx] ?? {}) }));
}

export function createSandboxGameState(overrides: Partial<GameState> = {}): GameState {
  const board = createEmptySquareBoard(8);
  // Single stack at (0,0) for selection tests
  const key = '0,0';
  (board.stacks as Map<string, any>).set(key, {
    rings: [1],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 1,
    position: { x: 0, y: 0 },
  });

  const players = createPlayers(overrides.players as Player[] | undefined);

  const base: GameState = {
    id: 'sandbox-1',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
  };

  return { ...base, ...overrides, players: overrides.players ?? players };
}

export function createLocalConfig(overrides: Partial<LocalConfig> = {}): LocalConfig {
  const base: LocalConfig = {
    numPlayers: 2,
    boardType: 'square8',
    playerTypes: ['human', 'human', 'ai', 'ai'],
    aiDifficulties: [5, 5, 5, 5],
  };
  return { ...base, ...overrides };
}

export function setIsMobile(value: boolean) {
  const mod = require('../../../src/client/hooks/useIsMobile') as {
    useIsMobile: jest.Mock;
  };
  mod.useIsMobile.mockReturnValue(value);
}

export function createMockSandboxContext(overrides: Partial<any> = {}): any {
  const config = createLocalConfig();
  const base = {
    config,
    setConfig: jest.fn(),
    isConfigured: false,
    setIsConfigured: jest.fn(),
    backendSandboxError: null,
    setBackendSandboxError: jest.fn(),
    sandboxEngine: null,
    sandboxPendingChoice: null,
    setSandboxPendingChoice: jest.fn(),
    sandboxCaptureChoice: null,
    setSandboxCaptureChoice: jest.fn(),
    sandboxCaptureTargets: [] as Position[],
    setSandboxCaptureTargets: jest.fn(),
    sandboxLastProgressAt: null as number | null,
    setSandboxLastProgressAt: jest.fn(),
    sandboxStallWarning: null as string | null,
    setSandboxStallWarning: jest.fn(),
    sandboxStateVersion: 0,
    setSandboxStateVersion: jest.fn(),
    sandboxDiagnosticsEnabled: true,
    developerToolsEnabled: false,
    setDeveloperToolsEnabled: jest.fn(),
    initLocalSandboxEngine: jest.fn(),
    getSandboxGameState: jest.fn(() => null),
    resetSandboxEngine: jest.fn(),
  };

  const merged = { ...base, ...overrides };

  if (merged.sandboxEngine) {
    const engine = merged.sandboxEngine;
    merged.sandboxEngine = {
      ...engine,
      getLpsTrackingState:
        typeof engine.getLpsTrackingState === 'function'
          ? engine.getLpsTrackingState
          : jest.fn(() => null),
      getValidMoves:
        typeof engine.getValidMoves === 'function' ? engine.getValidMoves : jest.fn(() => []),
      getChainCaptureContextForCurrentPlayer:
        typeof engine.getChainCaptureContextForCurrentPlayer === 'function'
          ? engine.getChainCaptureContextForCurrentPlayer
          : jest.fn(() => null),
    };
  }

  return merged;
}

// ─────────────────────────────────────────────────────────────────────────────
// DOM Helpers
// ─────────────────────────────────────────────────────────────────────────────

export function getSquareCell(x: number, y: number): HTMLButtonElement {
  const board = screen.getByTestId('board-view');
  const cell = board.querySelector<HTMLButtonElement>(`button[data-x="${x}"][data-y="${y}"]`);
  if (!cell) {
    throw new Error(`Failed to find sandbox board cell at (${x}, ${y})`);
  }
  return cell;
}
