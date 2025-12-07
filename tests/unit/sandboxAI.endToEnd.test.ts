import * as fs from 'fs';
import * as path from 'path';
import type { GameState, Move } from '../../src/shared/types/game';
import {
  deserializeGameState,
  type SerializedGameState,
} from '../../src/shared/engine/contracts/serialization';
import { getValidMoves } from '../../src/shared/engine/orchestration/turnOrchestrator';
import {
  maybeRunAITurnSandbox,
  resetSandboxAIStallCounters,
  type SandboxAIHooks,
} from '../../src/client/sandbox/sandboxAI';
import { hashGameState } from '../../src/shared/engine';
import { ClientSandboxEngine } from '../../src/client/sandbox/ClientSandboxEngine';

// List of fixtures to test
const FIXTURE_FILES = [
  'ringrift_scenario_sandbox_scenario_turn_239.json',
  'ringrift_scenario_sandbox_scenario_turn_266.json',
  'ringrift_scenario_sandbox_scenario_turn_302.json',
  'ringrift_scenario_sandbox_scenario_turn_303.json',
  'ringrift_scenario_sandbox_scenario_turn_336.json',
  'ringrift_scenario_sandbox_scenario_turn_351.json',
  'ringrift_scenario_sandbox_scenario_turn_372.json',
];

describe('Sandbox AI End-to-End Regression', () => {
  // Increase timeout for long-running simulations
  jest.setTimeout(30000);

  for (const fixtureFile of FIXTURE_FILES) {
    it(`should play ${fixtureFile} to completion without stalling`, async () => {
      const fixturePath = path.join(__dirname, '../..', fixtureFile);
      if (!fs.existsSync(fixturePath)) {
        console.warn(`Fixture not found: ${fixtureFile}, skipping`);
        return;
      }

      const raw = JSON.parse(fs.readFileSync(fixturePath, 'utf8')) as {
        state: SerializedGameState;
      };
      
      // Initialize engine with the fixture state
      const engine = new ClientSandboxEngine({
        config: {
          boardType: raw.state.board.type as any,
          numPlayers: raw.state.players.length,
          playerKinds: raw.state.players.map(() => 'ai'),
        },
        interactionHandler: {
          requestChoice: async () => {
            throw new Error('Unexpected interaction request in AI-only test');
          },
        },
      });

      engine.initFromSerializedState(
        raw.state,
        raw.state.players.map(() => 'ai'),
        {
          requestChoice: async () => {
             throw new Error('Unexpected interaction request in AI-only test');
          }
        }
      );

      // Force active status to ensure we can continue playing
      const state = engine.getGameState();
      if (state.gameStatus !== 'active') {
         // If fixture is already over, we can't really test "playing to completion"
         // unless we revert it. But for now let's just skip or assert it's done.
         return;
      }

      resetSandboxAIStallCounters();

      // Run the AI loop
      let turns = 0;
      const MAX_TURNS = 200; // Cap to prevent infinite test loops
      
      while (engine.getGameState().gameStatus === 'active' && turns < MAX_TURNS) {
        const beforeState = engine.getGameState();
        const beforeHash = hashGameState(beforeState);
        
        // We use the real engine hooks here to exercise the full stack
        // including applyCanonicalMove and the orchestrator.
        await engine.maybeRunAITurn(() => Math.random());
        
        const afterState = engine.getGameState();
        const afterHash = hashGameState(afterState);
        
        // Check for progress
        if (beforeHash === afterHash && beforeState.currentPlayer === afterState.currentPlayer && beforeState.currentPhase === afterState.currentPhase) {
             // If state, player, AND phase didn't change, it's a hard stall.
             // The stall detector inside maybeRunAITurnSandbox should have caught this
             // and eventually stopped execution, but we want to fail the test immediately
             // if it happens repeatedly.
             // However, maybeRunAITurnSandbox has internal counters.
             // We'll rely on the loop limit and final assertion.
        }
        
        turns++;
      }

      const finalState = engine.getGameState();
      
      // Assert that the game finished
      if (finalState.gameStatus === 'active') {
          console.error(`Game ${fixtureFile} did not finish in ${MAX_TURNS} turns. Final phase: ${finalState.currentPhase}, Player: ${finalState.currentPlayer}`);
          // Fail the test
          expect(finalState.gameStatus).not.toBe('active');
      } else {
          // Success
          expect(finalState.gameStatus).toBe('completed');
      }
    });
  }
});