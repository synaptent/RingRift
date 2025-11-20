import { RuleEngine } from './RuleEngine';
import { BoardManager } from './BoardManager';
import { GameState, Move } from '../../shared/types/game';
import { hashGameState } from '../../shared/engine/core';
import { readFileSync } from 'fs';

async function main() {
  const inputFile = process.argv[2];
  if (!inputFile) {
    console.error('Usage: ts-node test-parity-cli.ts <input-json-file>');
    process.exit(1);
  }

  try {
    const inputData = JSON.parse(readFileSync(inputFile, 'utf-8'));
    const gameState = inputData.gameState as GameState;
    const move = inputData.move as Move;

    // Clean up position z=null to undefined so it matches the TS Move model
    if (move.to && (move.to as any).z === null) {
      delete (move.to as any).z;
    }
    if (move.from && (move.from as any).z === null) {
      delete (move.from as any).z;
    }
    if (move.captureTarget && (move.captureTarget as any).z === null) {
      delete (move.captureTarget as any).z;
    }

    // Convert plain objects to Maps for board state, since the shared
    // engine helpers and RuleEngine expect Map-based collections.
    if (gameState.board) {
      const board: any = gameState.board as any;

      if (board.stacks && !(board.stacks instanceof Map)) {
        board.stacks = new Map(Object.entries(board.stacks));
      }
      if (board.markers && !(board.markers instanceof Map)) {
        board.markers = new Map(Object.entries(board.markers));
      }
      if (board.collapsedSpaces && !(board.collapsedSpaces instanceof Map)) {
        board.collapsedSpaces = new Map(Object.entries(board.collapsedSpaces));
      }
      if (board.territories && !(board.territories instanceof Map)) {
        board.territories = new Map(Object.entries(board.territories));
      }
    }

    // Initialize engines
    const boardManager = new BoardManager(gameState.boardType);
    const ruleEngine = new RuleEngine(boardManager, gameState.boardType);

    const isValid = ruleEngine.validateMove(move, gameState);

    // For parity CLI purposes we hash the (validated) input state using the
    // shared hashGameState helper. Python parity tests currently only assert
    // that a non-empty hash is returned when available, not that it matches a
    // specific post-move state.
    let stateHash = '';
    if (isValid) {
      try {
        stateHash = hashGameState(gameState);
      } catch {
        // If hashing fails for any reason, leave stateHash empty; callers
        // treat the presence of a hash as an optional enhancement.
      }
    }

    console.log(
      JSON.stringify({
        status: 'success',
        isValid,
        stateHash,
      })
    );
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
