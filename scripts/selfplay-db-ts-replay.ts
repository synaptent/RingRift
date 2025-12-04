#!/usr/bin/env ts-node
/**
 * Diagnostic script: replay a recorded self-play game into the
 * ClientSandboxEngine and log per-move TS state summaries.
 *
 * Usage (from repo root):
 *
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/selfplay-db-ts-replay.ts \\
 *     --db /absolute/or/relative/path/to/games.db \\
 *     --game 7f031908-655b-49af-ad05-f330e9d07488
 *
 * This mirrors the /sandbox self-play replay path (SelfPlayBrowser +
 * SandboxGameHost) but runs headless under Node so you can compare the
 * TS engine’s state sequence against Python’s GameReplayDB.get_state_at_move.
 */

import * as path from 'path';

import { getSelfPlayGameService } from '../src/server/services/SelfPlayGameService';
import {
  ClientSandboxEngine,
  type SandboxConfig,
  type SandboxInteractionHandler,
} from '../src/client/sandbox/ClientSandboxEngine';
import type { BoardType, GameState, Move, Position } from '../src/shared/types/game';
import { hashGameState } from '../src/shared/engine';

interface CliArgs {
  dbPath: string;
  gameId: string;
}

function parseArgs(argv: string[]): CliArgs {
  let dbPath = '';
  let gameId = '';

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if ((arg === '--db' || arg === '--dbPath') && argv[i + 1]) {
      dbPath = argv[i + 1];
      i += 1;
    } else if ((arg === '--game' || arg === '--gameId') && argv[i + 1]) {
      gameId = argv[i + 1];
      i += 1;
    }
  }

  if (!dbPath || !gameId) {
    console.error(
      'Usage: TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/selfplay-db-ts-replay.ts ' +
        '--db /path/to/games.db --game <gameId>'
    );
    process.exit(1);
  }

  const resolvedDb = path.isAbsolute(dbPath) ? dbPath : path.resolve(process.cwd(), dbPath);

  return { dbPath: resolvedDb, gameId };
}

/**
 * Normalize a recorded move from the self-play database into the canonical
 * Move surface expected by the sandbox engine.
 *
 * This mirrors SelfPlayBrowser.normalizeRecordedMove but is defined locally
 * to avoid a React dependency in this Node script.
 */
function normalizeRecordedMove(rawMove: Move, fallbackMoveNumber: number): Move {
  const anyMove = rawMove as any;

  const type: Move['type'] =
    anyMove.type === 'forced_elimination' ? 'eliminate_rings_from_stack' : anyMove.type;

  const timestampRaw = anyMove.timestamp;
  const timestamp: Date =
    timestampRaw instanceof Date
      ? timestampRaw
      : typeof timestampRaw === 'string'
        ? new Date(timestampRaw)
        : new Date();

  const from: Position | undefined =
    anyMove.from && typeof anyMove.from === 'object' ? anyMove.from : undefined;

  const moveNumber =
    typeof anyMove.moveNumber === 'number' && Number.isFinite(anyMove.moveNumber)
      ? anyMove.moveNumber
      : fallbackMoveNumber;

  const thinkTime =
    typeof anyMove.thinkTime === 'number'
      ? anyMove.thinkTime
      : typeof anyMove.thinkTimeMs === 'number'
        ? anyMove.thinkTimeMs
        : 0;

  return {
    ...anyMove,
    type,
    from,
    timestamp,
    thinkTime,
    moveNumber,
  } as Move;
}

function summarizeState(label: string, state: GameState): Record<string, unknown> {
  return {
    label,
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
    gameStatus: state.gameStatus,
    moveHistoryLength: state.moveHistory.length,
    stateHash: hashGameState(state),
  };
}

async function main(): Promise<void> {
  const { dbPath, gameId } = parseArgs(process.argv.slice(2));

  const service = getSelfPlayGameService();
  const detail = service.getGame(dbPath, gameId);

  if (!detail) {
    console.error(`Game ${gameId} not found in DB ${dbPath}`);
    process.exit(1);
    return;
  }

  // Sanitize initial state in the same way SelfPlayBrowser does before
  // passing it into /sandbox, so this script matches sandbox behaviour.
  const rawState = detail.initialState as any;
  const sanitizedState = rawState && typeof rawState === 'object' ? { ...rawState } : rawState;
  if (sanitizedState && Array.isArray(sanitizedState.moveHistory)) {
    sanitizedState.moveHistory = [];
  }
  if (sanitizedState && Array.isArray(sanitizedState.history)) {
    sanitizedState.history = [];
  }

  const config: SandboxConfig = {
    boardType: detail.boardType as BoardType,
    numPlayers: detail.numPlayers,
    playerKinds: Array.from({ length: detail.numPlayers }, () => 'human'),
  };

  const interactionHandler: SandboxInteractionHandler = {
    async requestChoice(choice: any) {
      const options = (choice?.options as any[]) ?? [];
      const selectedOption = options.length > 0 ? options[0] : undefined;
      return {
        choiceId: choice.id,
        playerNumber: choice.playerNumber,
        choiceType: choice.type,
        selectedOption,
      } as any;
    },
  };

  const engine = new ClientSandboxEngine({
    config,
    interactionHandler,
    traceMode: false,
  });

  engine.initFromSerializedState(sanitizedState, config.playerKinds, interactionHandler);

  const recordedMoves: Move[] = detail.moves.map((m) =>
    normalizeRecordedMove(m.move as Move, m.moveNumber)
  );

  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify({
      kind: 'ts-replay-initial',
      dbPath,
      gameId,
      totalRecordedMoves: recordedMoves.length,
      summary: summarizeState('initial', engine.getGameState()),
    })
  );

  let applied = 0;
  for (const move of recordedMoves) {
    applied += 1;

    await engine.applyCanonicalMoveForReplay(move);
    const state = engine.getGameState();
    // eslint-disable-next-line no-console
    console.log(
      JSON.stringify({
        kind: 'ts-replay-step',
        k: applied,
        moveType: move.type,
        movePlayer: move.player,
        moveNumber: move.moveNumber,
        summary: summarizeState(`after_move_${applied}`, state),
      })
    );
  }

  const finalState = engine.getGameState();
  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify({
      kind: 'ts-replay-final',
      appliedMoves: applied,
      summary: summarizeState('final', finalState),
    })
  );
}

void main();
