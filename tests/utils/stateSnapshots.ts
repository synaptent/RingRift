import { GameState, BoardState, Player, positionToString } from '../../src/shared/types/game';

export interface ComparableSnapshot {
  /** Arbitrary label for logging (e.g. 'backend-step-12'). */
  label: string;
  boardType: GameState['boardType'];
  currentPlayer: number;
  currentPhase: GameState['currentPhase'];
  gameStatus: GameState['gameStatus'];
  totalRingsInPlay: number;
  totalRingsEliminated: number;
  players: Array<{
    playerNumber: number;
    type: Player['type'];
    ringsInHand: number;
    eliminatedRings: number;
    territorySpaces: number;
  }>;
  stacks: Array<{
    key: string;
    controllingPlayer: number;
    stackHeight: number;
    capHeight: number;
    rings: number[];
  }>;
  markers: Array<{
    key: string;
    player: number;
  }>;
  collapsedSpaces: Array<{
    key: string;
    player: number;
  }>;
}

/**
 * Create a JSON-serialisable, order-stable snapshot of a full GameState.
 * This is used by parity tests to compare backend vs sandbox after each move.
 */
export function snapshotFromGameState(label: string, state: GameState): ComparableSnapshot {
  return {
    label,
    boardType: state.boardType,
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
    gameStatus: state.gameStatus,
    totalRingsInPlay: state.totalRingsInPlay,
    totalRingsEliminated: state.totalRingsEliminated,
    players: normalisePlayers(state.players),
    stacks: normaliseStacks(state.board),
    markers: normaliseMarkers(state.board),
    collapsedSpaces: normaliseCollapsedSpaces(state.board),
  };
}

function normalisePlayers(players: Player[]): ComparableSnapshot['players'] {
  return players
    .map((p) => ({
      playerNumber: p.playerNumber,
      type: p.type,
      ringsInHand: p.ringsInHand,
      eliminatedRings: p.eliminatedRings,
      territorySpaces: p.territorySpaces,
    }))
    .sort((a, b) => a.playerNumber - b.playerNumber);
}

function normaliseStacks(board: BoardState): ComparableSnapshot['stacks'] {
  const entries: ComparableSnapshot['stacks'] = [];

  for (const [key, stack] of board.stacks.entries()) {
    entries.push({
      key,
      controllingPlayer: stack.controllingPlayer,
      stackHeight: stack.stackHeight,
      capHeight: stack.capHeight,
      rings: [...stack.rings],
    });
  }

  entries.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  return entries;
}

function normaliseMarkers(board: BoardState): ComparableSnapshot['markers'] {
  const entries: ComparableSnapshot['markers'] = [];

  for (const [key, marker] of board.markers.entries()) {
    entries.push({
      key,
      player: marker.player,
    });
  }

  entries.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  return entries;
}

function normaliseCollapsedSpaces(board: BoardState): ComparableSnapshot['collapsedSpaces'] {
  const entries: ComparableSnapshot['collapsedSpaces'] = [];

  for (const [key, player] of board.collapsedSpaces.entries()) {
    entries.push({
      key,
      player,
    });
  }

  entries.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  return entries;
}

/**
 * Simple deep equality using JSON serialisation. This is sufficient for test
 * snapshots because ComparableSnapshot is already order-stable. The `label`
 * field is intentionally ignored so callers can use different labels (e.g.
 * 'backend-step-12' vs 'sandbox-step-12') without affecting equality.
 */
export function snapshotsEqual(a: ComparableSnapshot, b: ComparableSnapshot): boolean {
  const { label: _aLabel, ...restA } = a as any;
  const { label: _bLabel, ...restB } = b as any;
  return JSON.stringify(restA) === JSON.stringify(restB);
}

/**
 * Produce a minimal structured diff between two snapshots for logging.
 * This is intentionally shallow and focused on the most parity-relevant fields.
 */
export function diffSnapshots(
  a: ComparableSnapshot,
  b: ComparableSnapshot
): Record<string, unknown> {
  const diff: Record<string, unknown> = {};

  if (a.boardType !== b.boardType) {
    diff.boardType = { a: a.boardType, b: b.boardType };
  }
  if (a.currentPlayer !== b.currentPlayer) {
    diff.currentPlayer = { a: a.currentPlayer, b: b.currentPlayer };
  }
  if (a.currentPhase !== b.currentPhase) {
    diff.currentPhase = { a: a.currentPhase, b: b.currentPhase };
  }
  if (a.gameStatus !== b.gameStatus) {
    diff.gameStatus = { a: a.gameStatus, b: b.gameStatus };
  }
  if (a.totalRingsInPlay !== b.totalRingsInPlay) {
    diff.totalRingsInPlay = { a: a.totalRingsInPlay, b: b.totalRingsInPlay };
  }
  if (a.totalRingsEliminated !== b.totalRingsEliminated) {
    diff.totalRingsEliminated = { a: a.totalRingsEliminated, b: b.totalRingsEliminated };
  }

  if (JSON.stringify(a.players) !== JSON.stringify(b.players)) {
    diff.players = { a: a.players, b: b.players };
  }
  if (JSON.stringify(a.stacks) !== JSON.stringify(b.stacks)) {
    diff.stacks = { a: a.stacks, b: b.stacks };
  }
  if (JSON.stringify(a.markers) !== JSON.stringify(b.markers)) {
    diff.markers = { a: a.markers, b: b.markers };
  }
  if (JSON.stringify(a.collapsedSpaces) !== JSON.stringify(b.collapsedSpaces)) {
    diff.collapsedSpaces = { a: a.collapsedSpaces, b: b.collapsedSpaces };
  }

  return diff;
}

/**
 * Convenience helper for snapshotting a board+players pair directly, for
 * territory/capture micro-tests that do not construct a full GameState.
 */
export function snapshotFromBoardAndPlayers(
  label: string,
  board: BoardState,
  players: Player[]
): ComparableSnapshot {
  const totalRingsInPlay =
    players.reduce((sum, player) => sum + player.ringsInHand, 0) +
    Array.from(board.stacks.values()).reduce((sum, stack) => sum + stack.rings.length, 0);
  const fakeState: GameState = {
    id: 'snapshot',
    boardType: board.type,
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: players[0]?.playerNumber ?? 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 0, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(0),
    lastMoveAt: new Date(0),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay,
    totalRingsEliminated: 0,
    victoryThreshold: 0,
    territoryVictoryThreshold: 0,
  };

  return snapshotFromGameState(label, fakeState);
}
