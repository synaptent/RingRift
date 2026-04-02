/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');

// Simple JS wrapper that reconstructs Maps/Sets and calls the compiled JS
// globalActions helpers to avoid ts-node/ESM friction.
const {
  computeGlobalLegalActionsSummary,
  hasAnyGlobalMovementOrCapture,
} = require('../../dist/server/shared/engine/globalActions');

function revivePosition(pos) {
  return { x: pos.x, y: pos.y, z: pos.z ?? null };
}

function reviveBoard(raw) {
  const stacks = new Map();
  Object.entries(raw.stacks || {}).forEach(([k, v]) => {
    stacks.set(k, {
      position: revivePosition(v.position),
      rings: v.rings,
      stackHeight: v.stackHeight,
      capHeight: v.capHeight,
      controllingPlayer: v.controllingPlayer,
    });
  });

  const markers = new Map();
  Object.entries(raw.markers || {}).forEach(([k, v]) => {
    markers.set(k, { ...v, position: revivePosition(v.position) });
  });

  const collapsedSpaces = new Map();
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

function revivePlayers(rawPlayers) {
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

function inferTotalRingsInPlay(players, board) {
  const ringsInHand = players.reduce((sum, player) => sum + player.ringsInHand, 0);
  const ringsOnBoard = Array.from(board.stacks.values()).reduce(
    (sum, stack) => sum + stack.rings.length,
    0,
  );
  return ringsInHand + ringsOnBoard;
}

function main() {
  const bundlePath = process.argv[2];
  if (!bundlePath) {
    console.error('Usage: node scripts/debug/globalActionsFromBundle.js <bundle.json>');
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
  const totalRingsInPlay = inferTotalRingsInPlay(players, board);
  const state = {
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
    totalRingsInPlay,
    totalRingsEliminated: tsState.totalRingsEliminated ?? 0,
    victoryThreshold: tsState.victoryThreshold ?? 0,
    territoryVictoryThreshold: tsState.territoryVictoryThreshold ?? 0,
  };

  const summaries = state.players.map((p) => {
    const summary = computeGlobalLegalActionsSummary(state, p.playerNumber);
    const hasMoveCapture = hasAnyGlobalMovementOrCapture(state, p.playerNumber);
    return { player: p.playerNumber, summary, hasMoveCapture };
  });
  console.log(JSON.stringify({ currentPhase: state.currentPhase, summaries }, null, 2));
}

main();
