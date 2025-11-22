import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Position,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString,
} from '../../src/shared/types/game';
import { addStack, pos } from '../utils/fixtures';
import { movementRuleScenarios, MovementRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → ClientSandboxEngine movement scenarios
 *
 * Mirrors Section 8.2–8.3 / FAQ 2–3 movement examples from
 * `movementRuleScenarios` against the client-local sandbox engine.
 * This is intentionally parallel to RulesMatrix.Movement.RuleEngine tests
 * but uses ClientSandboxEngine.getValidLandingPositionsForCurrentPlayer
 * instead of RuleEngine.getValidMoves.
 */

describe('RulesMatrix → ClientSandboxEngine movement scenarios (Section 8.2–8.3; FAQ 2–3)', () => {
  function createEngine(boardType: BoardType): { engine: ClientSandboxEngine; state: GameState } {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      // Movement scenarios should not surface choices, but we provide a
      // trivial handler for completeness.
      async requestChoice<TChoice>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as PlayerChoiceResponseFor<any>;
      },
    };

    const engine = new ClientSandboxEngine({ config, interactionHandler: handler });
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;
    return { engine, state };
  }

  // For now we exercise the core minimum-distance and marker-landing scenarios
  // on the sandbox side. The more intricate blocking example
  // `Rules_8_3_Q3_blocked_by_stacks_and_collapsed_square8` remains asserted
  // via the backend RulesMatrix + unit suites.
  const scenarios: MovementRuleScenario[] = movementRuleScenarios.filter(
    (s) => s.ref.id !== 'Rules_8_3_Q3_blocked_by_stacks_and_collapsed_square8'
  );

  test.each<MovementRuleScenario>(scenarios)(
    '%s → sandbox movement matches rules/FAQ expectations',
    (scenario) => {
      const { engine, state } = createEngine(scenario.boardType as BoardType);
      const board = state.board;

      state.currentPlayer = 1;

      const { origin, stackHeight, blockers } = scenario;
      const originPos: Position =
        origin.z != null ? pos(origin.x, origin.y, origin.z) : pos(origin.x, origin.y);

      // Place the moving stack for Player 1.
      addStack(board, originPos, 1, stackHeight);

      // Optional blockers: stacks and collapsed spaces.
      if (blockers) {
        for (const blocker of blockers) {
          const blockerPos: Position =
            blocker.position.z != null
              ? pos(blocker.position.x, blocker.position.y, blocker.position.z as any)
              : pos(blocker.position.x, blocker.position.y);

          if (blocker.type === 'stack') {
            const h = blocker.height ?? 1;
            const owner = blocker.controllingPlayer ?? 2;
            addStack(board, blockerPos, owner, h);
          } else if (blocker.type === 'collapsed') {
            const key = positionToString(blockerPos);
            board.collapsedSpaces.set(key, 0);
          }
        }
      }

      // Additional marker scenarios mirror the backend RulesMatrix movement tests.
      if (scenario.ref.id === 'Rules_8_2_Q2_markers_any_valid_space_beyond_square8') {
        if (scenario.boardType !== 'square8') {
          throw new Error('Rules_8_2_Q2_markers_any_valid_space_beyond_square8 must use square8');
        }
        const marker1: Position = { x: originPos.x + 1, y: originPos.y };
        const marker2: Position = { x: originPos.x + 2, y: originPos.y };

        board.markers.set(positionToString(marker1), {
          player: 2,
          position: marker1,
          type: 'regular',
        });
        board.markers.set(positionToString(marker2), {
          player: 2,
          position: marker2,
          type: 'regular',
        });
      } else if (scenario.ref.id === 'Rules_8_2_Q2_marker_landing_own_vs_opponent_square8') {
        if (scenario.boardType !== 'square8') {
          throw new Error('Rules_8_2_Q2_marker_landing_own_vs_opponent_square8 must use square8');
        }

        const ownMarker: Position = { x: originPos.x + 2, y: originPos.y };
        const oppMarker: Position = { x: originPos.x, y: originPos.y + 2 };

        board.markers.set(positionToString(ownMarker), {
          player: 1,
          position: ownMarker,
          type: 'regular',
        });
        board.markers.set(positionToString(oppMarker), {
          player: 2,
          position: oppMarker,
          type: 'regular',
        });
      } else if (scenario.ref.id === 'Rules_8_2_Q2_marker_landing_own_vs_opponent_hexagonal') {
        if (scenario.boardType !== 'hexagonal') {
          throw new Error(
            'Rules_8_2_Q2_marker_landing_own_vs_opponent_hexagonal must use hexagonal'
          );
        }
        const originCube: Position =
          origin.z != null ? originPos : ({ x: origin.x, y: origin.y, z: 0 } as any);

        const ownMarker: Position = {
          x: originCube.x + 2,
          y: originCube.y - 2,
          z: originCube.z,
        };
        const oppMarker: Position = {
          x: originCube.x - 2,
          y: originCube.y + 2,
          z: originCube.z,
        };

        board.markers.set(positionToString(ownMarker), {
          player: 1,
          position: ownMarker,
          type: 'regular',
        });
        board.markers.set(positionToString(oppMarker), {
          player: 2,
          position: oppMarker,
          type: 'regular',
        });
      }

      const landings = engine.getValidLandingPositionsForCurrentPlayer(originPos);

      if (scenario.ref.id.startsWith('Rules_8_2_Q2_minimum_distance')) {
        // Minimum distance invariant: there must be no landing closer than stackHeight.
        expect(landings.length).toBeGreaterThan(0);

        const hasTooShort = landings.some((to) => {
          if (scenario.boardType === 'hexagonal') {
            const fromCube =
              origin.z != null
                ? originPos
                : ({ x: origin.x, y: origin.y, z: -origin.x - origin.y } as any);
            const toCube =
              (to as any).z != null ? (to as any) : ({ x: to.x, y: to.y, z: -to.x - to.y } as any);

            const dx = toCube.x - fromCube.x;
            const dy = toCube.y - fromCube.y;
            const dz = toCube.z - fromCube.z;
            const dist = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
            return dist < stackHeight;
          }

          const dx = Math.abs(to.x - originPos.x);
          const dy = Math.abs(to.y - originPos.y);
          const dist = Math.max(dx, dy);
          return dist < stackHeight;
        });

        expect(hasTooShort).toBe(false);
      } else if (scenario.ref.id === 'Rules_8_2_Q2_markers_any_valid_space_beyond_square8') {
        const landingKeys = landings.map((p) => positionToString(p));

        const landing1 = positionToString({ x: originPos.x + 3, y: originPos.y });
        const landing2 = positionToString({ x: originPos.x + 4, y: originPos.y });

        expect(landingKeys).toContain(landing1);
        expect(landingKeys).toContain(landing2);
      } else if (
        scenario.ref.id === 'Rules_8_2_Q2_marker_landing_own_vs_opponent_square8' ||
        scenario.ref.id === 'Rules_8_2_Q2_marker_landing_own_vs_opponent_hexagonal'
      ) {
        const landingKeys = landings.map((p) => positionToString(p));

        if (scenario.boardType === 'square8') {
          const ownKey = positionToString({ x: originPos.x + 2, y: originPos.y });
          const oppKey = positionToString({ x: originPos.x, y: originPos.y + 2 });

          expect(landingKeys).toContain(ownKey);
          expect(landingKeys).not.toContain(oppKey);
        } else if (scenario.boardType === 'hexagonal') {
          const fromCube =
            origin.z != null
              ? originPos
              : ({ x: origin.x, y: origin.y, z: -origin.x - origin.y } as any);
          const ownKey = positionToString({
            x: fromCube.x + 2,
            y: fromCube.y - 2,
            z: fromCube.z,
          });
          const oppKey = positionToString({
            x: fromCube.x - 2,
            y: fromCube.y + 2,
            z: fromCube.z,
          });

          expect(landingKeys).toContain(ownKey);
          expect(landingKeys).not.toContain(oppKey);
        } else {
          throw new Error(
            `Unsupported boardType for marker landing scenario: ${scenario.boardType as string}`
          );
        }
      } else if (scenario.ref.id === 'Rules_8_3_Q3_blocked_by_stacks_and_collapsed_square8') {
        // Blocking invariant: no landing may "jump through" the blocking stack
        // or collapsed space along their rays.
        const stackBlock = blockers!.find((b) => b.type === 'stack')!;
        const collapsedBlock = blockers!.find((b) => b.type === 'collapsed')!;

        const illegalThroughStack = landings.some((to) => {
          return to.y === origin.y && to.x > (stackBlock.position.x ?? 0);
        });

        const illegalThroughCollapsed = landings.some((to) => {
          return to.x === origin.x && to.y > (collapsedBlock.position.y ?? 0);
        });

        expect(illegalThroughStack).toBe(false);
        expect(illegalThroughCollapsed).toBe(false);
      } else {
        // If a new MovementRuleScenario is added without explicit logic here,
        // fail loudly so the test can be extended alongside the matrix.
        throw new Error(`Unhandled MovementRuleScenario id: ${scenario.ref.id}`);
      }
    }
  );
});
