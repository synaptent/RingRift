/* eslint-disable no-console */
import fs from 'fs';
import path from 'path';
import type { GameState, BoardState, Player, Position } from '../src/shared/types/game';
import {
  computeGlobalLegalActionsSummary,
  hasAnyGlobalMovementOrCapture,
} from '../src/shared/engine/globalActions';

function revivePosition(pos: any): Position {
  return { x: pos.x, y: pos.y, z: pos.z ?? null };
}

function reviveBoard(raw: any): BoardState {
  const stacks = new Map<string, any>();
  Object.entries(raw.stacks || {}).forEach(([k, v]) => {
    stacks.set(k, {
      position: revivePosition((v as any).position),
      rings: (v as any).rings,
      stackHeight: (v as any).stackHeight,
      capHeight: (v as any).capHeight,
      controllingPlayer: (v as any).controllingPlayer,
    });
  });

  const markers = new Map<string, any>();
  Object.entries(raw.markers || {}).forEach(([k, v]) => {
    markers.set(k, { ...v, position: revivePosition((v as any).position) });
  });

  const collapsedSpaces = new Map<string, number>();
  Object.entries(raw.collapsedSpaces || {}).forEach(([k, v]) => {
    collapsedSpaces.set(k, Number(v));
  });

  return {
    stacks,
    markers,
    collapsedSpaces,
    territories: new Map(),
    formedLines: raw.formedLines || [],
    eliminatedRings: raw.eliminatedRings || {},
    size: raw.size,
    type: raw.type,
  };
}

function revivePlayers(rawPlayers: any[]): Player[] {
  return rawPlayers.map((p) => ({
    id: String(p.id ?? p.playerNumber),
    username: `P${p.playerNumber}`,
    type: 'ai',
    playerNumber: p.playerNumber,
    isReady: true,
    timeRemaining: 0,
    ringsInHand: p.ringsInHand,
    eliminatedRings: p.eliminatedRings,
    territorySpaces: p.territorySpaces ?? 0,
  }));
}

function main() {
  const bundlePath = process.argv[2];
  if (!bundlePath) {
    console.error('Usage: ts-node scripts/debug/globalActionsFromBundle.ts <bundle.json>');
    process.exit(1);
  }
  const bundle = JSON.parse(fs.readFileSync(bundlePath, 'utf8'));
  const tsState = bundle.ts_states?.['149'] ?? bundle.ts_states?.[bundle.ts_k_values?.[1]];
  if (!tsState) {
    console.error('Could not find ts state at k=149 in bundle');
    process.exit(1);
  }
  const board = reviveBoard(tsState.board);
  const players = revivePlayers(tsState.players);
  const state: GameState = {
    ...tsState,
    board,
    players,
    moveHistory: [],
    history: [],
    spectators: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    id: tsState.gameId ?? 'debug',
    boardType: tsState.board.type,
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 0,
    totalRingsEliminated: tsState.totalRingsEliminated ?? 0,
    victoryThreshold: tsState.victoryThreshold ?? 0,
    territoryVictoryThreshold: tsState.territoryVictoryThreshold ?? 0,
  } as GameState;

  const summaries = state.players.map((p) => {
    const summary = computeGlobalLegalActionsSummary(state, p.playerNumber);
    const hasMoveCapture = hasAnyGlobalMovementOrCapture(state, p.playerNumber);
    return { player: p.playerNumber, summary, hasMoveCapture };
  });
  console.log(JSON.stringify({ currentPhase: state.currentPhase, summaries }, null, 2));
}

main();
