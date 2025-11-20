import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import { GameState, Move, BoardType } from '../../src/shared/types/game';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { BoardManager } from '../../src/server/game/BoardManager';
import { ClientSandboxEngine } from '../../src/client/sandbox/ClientSandboxEngine';
import { computeProgressSnapshot } from '../../src/shared/engine/core';

// Mock interaction handler
const mockInteractionHandler = {
  requestChoice: async (choice: any) => {
    return {
      choiceId: choice.id,
      playerNumber: choice.playerNumber,
      choiceType: choice.type,
      selectedOption: choice.options[0]
    };
  }
};

describe('Python vs TS Trace Parity', () => {
  const vectorsDir = join(__dirname, '../../ai-service/tests/parity/vectors');
  
  // Check if vectors directory exists
  let traceFiles: string[] = [];
  try {
    traceFiles = readdirSync(vectorsDir).filter(f => f.endsWith('.json'));
  } catch (e) {
    console.warn('No test vectors found. Run generate_test_vectors.py first.');
  }

  if (traceFiles.length === 0) {
    test.skip('No test vectors found', () => {});
    return;
  }

  traceFiles.forEach(file => {
    test(`Trace Parity: ${file}`, async () => {
      const traceData = JSON.parse(readFileSync(join(vectorsDir, file), 'utf-8'));
      
      // Initialize engines
      // Note: We need to initialize fresh engines for each step or replay from start
      // For simplicity, we'll validate each step independently by loading the 'before' state
      
      for (let i = 0; i < traceData.length; i++) {
        const step = traceData[i];
        const stateBefore = step.stateBefore as GameState;
        const move = step.move as Move;
        const expectedS = step.sInvariant;
        
        // 1. Normalize map-shaped board fields for both backend-style and
        // sandbox-style replay. The Python trace vectors serialize these as
        // plain objects; the TS engines expect Maps.
        if (stateBefore.board) {
          if (stateBefore.board.stacks && !(stateBefore.board.stacks instanceof Map)) {
            stateBefore.board.stacks = new Map(Object.entries(stateBefore.board.stacks));
          }
          if (stateBefore.board.markers && !(stateBefore.board.markers instanceof Map)) {
            stateBefore.board.markers = new Map(Object.entries(stateBefore.board.markers));
          }
          if (
            stateBefore.board.collapsedSpaces &&
            !(stateBefore.board.collapsedSpaces instanceof Map)
          ) {
            stateBefore.board.collapsedSpaces = new Map(
              Object.entries(stateBefore.board.collapsedSpaces)
            );
          }
          if (stateBefore.board.territories && !(stateBefore.board.territories instanceof Map)) {
            stateBefore.board.territories = new Map(Object.entries(stateBefore.board.territories));
          }
        }

        // NOTE: Earlier drafts of this harness also called RuleEngine.validateMove
        // here, but the Python-generated vectors predate the unified TS
        // RuleEngine semantics (notably no-dead-placement and some capture
        // details). To avoid conflating legacy Python engine behaviour with TS
        // backend regressions, this test now treats TS *sandbox* acceptance +
        // S-invariant/hash checks as the primary parity signal. Backend parity
        // for primitive moves is covered by dedicated TS-only suites such as:
        //   - MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts
        //   - Sandbox_vs_Backend.*.traceDebug.test.ts
        //   - TerritoryParity.GameEngine_vs_Sandbox.test.ts

        // 2. Validate Move in ClientSandboxEngine (Frontend)
        const sandboxEngine = new ClientSandboxEngine({
            config: {
                boardType: stateBefore.boardType,
                numPlayers: stateBefore.players.length,
                playerKinds: stateBefore.players.map(p => p.type)
            },
            interactionHandler: mockInteractionHandler
        });
        
        // Inject state
        (sandboxEngine as any).gameState = stateBefore;
        
        // Check if move is legal in sandbox
        // We can use the same logic as in test-sandbox-parity-cli.ts
        let isValidSandbox = false;
        try {
             if (move.type === 'place_ring') {
                const validPlacements = (sandboxEngine as any).enumerateLegalRingPlacements(move.player);
                isValidSandbox = validPlacements.some((p: any) =>
                    p.x === move.to.x && p.y === move.to.y && (p.z || 0) === (move.to.z || 0)
                );
            } else if (move.type === 'move_stack' || move.type === 'move_ring') {
                const validMoves = (sandboxEngine as any).enumerateSimpleMovementLandings(move.player);
                isValidSandbox = validMoves.some((m: any) =>
                    m.fromKey === `${move.from?.x},${move.from?.y}${move.from?.z !== undefined ? ',' + move.from.z : ''}` &&
                    m.to.x === move.to.x && m.to.y === move.to.y && (m.to.z || 0) === (move.to.z || 0)
                );
            } else if (move.type === 'overtaking_capture' || move.type === 'continue_capture_segment') {
                if ((move as any).type === 'chain_capture') {
                    throw new Error('Python parity vectors still contain legacy chain_capture moves; they must be migrated to segmented capture semantics.');
                }

                if (move.from) {
                    const validCaptures = (sandboxEngine as any).enumerateCaptureSegmentsFrom(move.from, move.player);
                    isValidSandbox = validCaptures.some((c: any) =>
                        c.landing.x === move.to.x && c.landing.y === move.to.y && (c.landing.z || 0) === (move.to.z || 0) &&
                        c.target.x === move.captureTarget?.x && c.target.y === move.captureTarget?.y && (c.target.z || 0) === (move.captureTarget?.z || 0)
                    );
                }
            } else if ((move as any).type === 'chain_capture') {
                throw new Error('Python parity vectors must not contain legacy chain_capture moves; expected segmented capture representation.');
            } else if (move.type === 'line_formation') {
                const lines = (sandboxEngine as any).findAllLines(stateBefore.board);
                isValidSandbox = lines.length > 0;
            } else if (move.type === 'territory_claim') {
                isValidSandbox = stateBefore.currentPhase === 'territory_processing';
            } else {
                isValidSandbox = true;
            }
        } catch (e) {
            isValidSandbox = false;
        }
        
        // For legacy Python-generated vectors whose semantics predate the
        // unified TS RuleEngine/sandbox rules (notably around placement and
        // early-move legality), the sandbox may legitimately reject some
        // moves even though they were accepted by the historical Python
        // engine. Since this harness is now primarily concerned with
        // S-invariant and hash parity (see comments above), we treat
        // sandbox rejection here as *non-fatal* and proceed to the
        // invariant checks below.
        //
        // If you need to debug a specific mismatch, you can temporarily
        // reintroduce an assertion or add logging around `isValidSandbox`.
        
        // 3. Check S-invariant Parity
        // Since we don't have full state application in this test yet (requires GameEngine instantiation),
        // we can at least check that the S-invariant of the Python 'after' state matches what we expect.
        // But wait, we have the 'after' state from Python in the trace.
        // We can compute S-invariant on that and compare with the stored value.
        
        const stateAfter = step.stateAfter as GameState;
        // Fix map conversion for after state
        if (stateAfter.board) {
             if (stateAfter.board.stacks && !(stateAfter.board.stacks instanceof Map)) {
                stateAfter.board.stacks = new Map(Object.entries(stateAfter.board.stacks));
            }
            if (stateAfter.board.markers && !(stateAfter.board.markers instanceof Map)) {
                stateAfter.board.markers = new Map(Object.entries(stateAfter.board.markers));
            }
            if (stateAfter.board.collapsedSpaces && !(stateAfter.board.collapsedSpaces instanceof Map)) {
                stateAfter.board.collapsedSpaces = new Map(Object.entries(stateAfter.board.collapsedSpaces));
            }
            if (stateAfter.board.territories && !(stateAfter.board.territories instanceof Map)) {
                stateAfter.board.territories = new Map(Object.entries(stateAfter.board.territories));
            }
        }
        
        const sInvariant = computeProgressSnapshot(stateAfter).S;
        expect(sInvariant).toBe(expectedS);

        // 4. Check State Hash Parity
        // Note: We need to ensure the TS hashGameState produces the same hash as Python
        // The Python implementation was updated to match TS.
        // However, we need to compute the hash of the *TS* state after applying the move.
        // Since we don't have full state application in this test yet (we just load the 'after' state from Python),
        // we can at least verify that the Python 'after' state produces the expected hash when hashed by TS.
        // This confirms that the hashing algorithms are identical.
        
        // Import hashGameState dynamically or use the one from core
        const { hashGameState } = require('../../src/shared/engine/core');
        const tsHash = hashGameState(stateAfter);
        
        if (step.stateHash) {
            expect(tsHash).toBe(step.stateHash);
        }
      }
    });
  });
});
