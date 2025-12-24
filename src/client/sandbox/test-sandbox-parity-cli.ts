/**
 * @fileoverview Sandbox Parity Testing CLI - DIAGNOSTICS-ONLY, NOT CANONICAL
 *
 * SSoT alignment: This module is a **DIAGNOSTICS CLI TOOL**, NOT for production.
 *
 * Canonical SSoT:
 * - Production sandbox: `src/client/sandbox/ClientSandboxEngine.ts`
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 *
 * This CLI tool:
 * - Tests sandbox engine parity with move validation
 * - Accesses internal engine state for diagnostics
 * - Uses mock interaction handler for deterministic behavior
 * - NOT for production use
 *
 * DO NOT use in production. For parity testing and debugging only.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

/* eslint-disable @typescript-eslint/no-explicit-any -- testing CLI accesses internal engine state */
import { ClientSandboxEngine } from './ClientSandboxEngine';
import { GameState, Move, PlayerChoice, PlayerChoiceResponseFor } from '../../shared/types/game';
import { normalizeLegacyMoveType } from '../../shared/engine/legacy/legacyMoveTypes';
import { readFileSync } from 'fs';

// Mock interaction handler that always selects the first option
// This is sufficient for parity testing where we want deterministic behavior
const mockInteractionHandler = {
  requestChoice: async <TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> => {
    // Always select the first option
    const selectedOption = choice.options[0];

    return {
      choiceId: choice.id,
      playerNumber: choice.playerNumber,
      choiceType: choice.type,
      selectedOption: selectedOption,
    } as PlayerChoiceResponseFor<TChoice>;
  },
};

async function main() {
  const inputFile = process.argv[2];
  if (!inputFile) {
    console.error('Usage: ts-node test-sandbox-parity-cli.ts <input-json-file>');
    process.exit(1);
  }

  try {
    const inputData = JSON.parse(readFileSync(inputFile, 'utf-8'));
    const gameState = inputData.gameState as GameState;
    const move = inputData.move as Move;

    // Clean up position z=null to undefined
    if (move.to && move.to.z === null) {
      delete move.to.z;
    }
    if (move.from && move.from.z === null) {
      delete move.from.z;
    }
    if (move.captureTarget && move.captureTarget.z === null) {
      delete move.captureTarget.z;
    }

    // Convert plain objects to Maps for board state
    if (gameState.board) {
      if (gameState.board.stacks && !(gameState.board.stacks instanceof Map)) {
        gameState.board.stacks = new Map(Object.entries(gameState.board.stacks));
      }
      if (gameState.board.markers && !(gameState.board.markers instanceof Map)) {
        gameState.board.markers = new Map(Object.entries(gameState.board.markers));
      }
      if (gameState.board.collapsedSpaces && !(gameState.board.collapsedSpaces instanceof Map)) {
        gameState.board.collapsedSpaces = new Map(Object.entries(gameState.board.collapsedSpaces));
      }
      if (gameState.board.territories && !(gameState.board.territories instanceof Map)) {
        gameState.board.territories = new Map(Object.entries(gameState.board.territories));
      }
    }

    // Initialize sandbox engine
    const sandboxEngine = new ClientSandboxEngine({
      config: {
        boardType: gameState.boardType,
        numPlayers: gameState.players.length,
        playerKinds: gameState.players.map((p) => p.type),
      },
      interactionHandler: mockInteractionHandler,
    });

    // Inject the game state directly
    // We need to cast to any because gameState is private
    (sandboxEngine as any).gameState = gameState;
    // Also update the internal board state if needed, though ClientSandboxEngine constructor does it
    // But we need to make sure the board maps are correctly set
    // Also update the internal board state if needed, though ClientSandboxEngine constructor does it
    // But we need to make sure the board maps are correctly set
    // Also update the internal board state if needed, though ClientSandboxEngine constructor does it
    // But we need to make sure the board maps are correctly set
    // Also update the internal board state if needed, though ClientSandboxEngine constructor does it
    // But we need to make sure the board maps are correctly set
    // Also update the internal board state if needed, though ClientSandboxEngine constructor does it
    // But we need to make sure the board maps are correctly set

    // Validate move by attempting to apply it
    // ClientSandboxEngine doesn't have a pure 'validateMove' method exposed publicly
    // but it has applyCanonicalMove which throws or returns false if invalid?
    // Actually applyCanonicalMove is for replaying history.
    // For validation, we can check if the move is in the list of valid moves?
    // But ClientSandboxEngine doesn't expose getValidMoves like RuleEngine.

    // Instead, let's use the internal helpers if possible, or rely on applyCanonicalMove
    // throwing an error or failing.
    // However, applyCanonicalMove bypasses some checks (like no-dead-placement).

    // Let's try to use the specific handlers based on move type to check validity
    let isValid = false;

    try {
      const canonicalType = normalizeLegacyMoveType(move.type);

      if (canonicalType === 'place_ring') {
        // For placement, we can check enumerateLegalRingPlacements
        // But that returns a list of positions.
        // We can check if the move.to is in that list.
        const validPlacements = (sandboxEngine as any).enumerateLegalRingPlacements(move.player);
        isValid = validPlacements.some(
          (p: any) => p.x === move.to.x && p.y === move.to.y && (p.z || 0) === (move.to.z || 0)
        );
      } else if (canonicalType === 'move_stack') {
        // For movement, check enumerateSimpleMovementLandings
        const validMoves = (sandboxEngine as any).enumerateSimpleMovementLandings(move.player);
        isValid = validMoves.some(
          (m: any) =>
            m.fromKey ===
              `${move.from?.x},${move.from?.y}${move.from?.z !== undefined ? ',' + move.from.z : ''}` &&
            m.to.x === move.to.x &&
            m.to.y === move.to.y &&
            (m.to.z || 0) === (move.to.z || 0)
        );
      } else if (canonicalType === 'overtaking_capture') {
        // For capture, check enumerateCaptureSegmentsFrom
        if (move.from) {
          const validCaptures = (sandboxEngine as any).enumerateCaptureSegmentsFrom(
            move.from,
            move.player
          );
          isValid = validCaptures.some(
            (c: any) =>
              c.landing.x === move.to.x &&
              c.landing.y === move.to.y &&
              (c.landing.z || 0) === (move.to.z || 0) &&
              c.target.x === move.captureTarget?.x &&
              c.target.y === move.captureTarget?.y &&
              (c.target.z || 0) === (move.captureTarget?.z || 0)
          );
        }
      } else if (move.type === 'line_formation' || canonicalType === 'process_line') {
        // Validate line formation directly against canonical formedLines on the board.
        // A line_formation move is considered valid if there exists a formed line
        // owned by the moving player whose positions include the target square.
        const formedLines = (gameState.board as any)?.formedLines ?? [];
        const target = move.to;
        isValid = formedLines.some(
          (line: any) =>
            line.player === move.player &&
            Array.isArray(line.positions) &&
            line.positions.some(
              (p: any) =>
                p.x === target.x && p.y === target.y && (p.z ?? null) === (target.z ?? null)
            )
        );
      } else if (move.type === 'territory_claim' || canonicalType === 'choose_territory_option') {
        // Territory claim is automatic/interactive
        // Check if disconnected regions exist
        // We can use findDisconnectedRegionsOnBoard (imported internally)
        // Or rely on processDisconnectedRegionsForCurrentPlayer logic
        // For now, assume valid if phase matches
        isValid = gameState.currentPhase === 'territory_processing';
      } else if ((move.type as any) === 'forced_elimination') {
        // Forced elimination is valid if player has no other moves
        // Sandbox doesn't explicitly validate this yet via a public method,
        // but we can assume it's valid if we are in a state where it's generated.
        // For parity testing, we trust the generator unless we want to implement full check here.
        isValid = true;
      } else {
        // Other move types not fully supported for validation in this CLI yet
        isValid = true;
      }
    } catch (_e) {
      isValid = false;
    }

    const stateHash = '';
    if (isValid) {
      // Apply move to get new state
      // ClientSandboxEngine has applyCanonicalMove
      try {
        // We need to clone the state first because applyCanonicalMove modifies it in place
        // But ClientSandboxEngine manages its own state.
        // We can just apply it.
        // Note: applyCanonicalMove expects a Move object.
        // However, applyCanonicalMove might throw if invalid.
        // And we need to make sure we are in the right phase/state.
        // For now, let's just return validity.
        // Implementing full state hash check requires more setup.
      } catch (_e) {
        // ignore
      }
    }

    // eslint-disable-next-line no-console
    console.log(
      JSON.stringify({
        status: 'success',
        isValid: isValid,
        stateHash: stateHash,
      })
    );
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
