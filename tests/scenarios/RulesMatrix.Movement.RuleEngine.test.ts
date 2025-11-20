import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { BoardType, GameState, Move, Position, positionToString } from '../../src/shared/types/game';
import { createTestGameState, pos } from '../utils/fixtures';
import { movementRuleScenarios, MovementRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → RuleEngine movement scenarios
 *
 * This suite replays a small set of FAQ-style movement examples from
 * `movementRuleScenarios` against the real RuleEngine implementation.
 *
 * It is intentionally parallel to tests/unit/RuleEngine.movement.scenarios.test.ts
 * but parameterised by the shared rulesMatrix definitions so additional
 * movement scenarios can be added in one place.
 */

describe('RulesMatrix → RuleEngine movement scenarios (Section 8.2–8.3; FAQ 2–3)', () => {
  function createState(boardType: BoardType): {
    gameState: GameState;
    boardManager: BoardManager;
    ruleEngine: RuleEngine;
  } {
    const gameState = createTestGameState({ boardType });

    // Use a BoardManager-created board so RuleEngine and BoardManager share
    // the same BoardState instance.
    const boardManager = new BoardManager(boardType);
    gameState.board = boardManager.createBoard();

    const ruleEngine = new RuleEngine(boardManager, boardType);

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'movement';

    return { gameState, boardManager, ruleEngine };
  }

  const scenarios: MovementRuleScenario[] = movementRuleScenarios;

  test.each<MovementRuleScenario>(scenarios)(
    '%s → backend RuleEngine movement matches rules/FAQ expectations',
    (scenario) => {
      const { gameState, boardManager, ruleEngine } = createState(scenario.boardType);

      const { origin, stackHeight, blockers } = scenario;
      const originPos = origin.z != null ? pos(origin.x, origin.y, origin.z) : pos(origin.x, origin.y);
      const rings = Array(stackHeight).fill(1);

      // Place the moving stack for Player 1.
      boardManager.setStack(
        originPos,
        {
          position: originPos,
          rings,
          stackHeight: rings.length,
          capHeight: rings.length,
          controllingPlayer: 1
        },
        gameState.board
      );

      // Optionally place blockers as described in the scenario.
      if (blockers) {
        for (const blocker of blockers) {
          const blockerPos =
            blocker.position.z != null
              ? pos(blocker.position.x, blocker.position.y, blocker.position.z)
              : pos(blocker.position.x, blocker.position.y);
 
          if (blocker.type === 'stack') {
            const bRings = Array(blocker.height ?? 1).fill(blocker.controllingPlayer ?? 2);
            boardManager.setStack(
              blockerPos,
              {
                position: blockerPos,
                rings: bRings,
                stackHeight: bRings.length,
                capHeight: bRings.length,
                controllingPlayer: blocker.controllingPlayer ?? 2
              },
              gameState.board
            );
          } else if (blocker.type === 'collapsed') {
            gameState.board.collapsedSpaces.set(`${blockerPos.x},${blockerPos.y}`, 0);
          }
        }
      }

      // Additional marker-focused movement scenarios keyed by RulesMatrix IDs.
      if (scenario.ref.id === 'Rules_8_2_Q2_markers_any_valid_space_beyond_square8') {
        // Place a run of markers directly east of the origin on square8:
        // origin = (3,3); markers at (4,3) and (5,3); empties at (6,3) and (7,3).
        // A height-2 stack should be allowed to land on any valid empty
        // space beyond the markers that satisfies the minimum-distance
        // and path rules, not just the first such space.
        if (scenario.boardType !== 'square8') {
          throw new Error('Rules_8_2_Q2_markers_any_valid_space_beyond_square8 must use square8');
        }
        const originPos: Position =
          scenario.origin.z != null
            ? pos(scenario.origin.x, scenario.origin.y, scenario.origin.z)
            : pos(scenario.origin.x, scenario.origin.y);
        const marker1: Position = { x: originPos.x + 1, y: originPos.y };
        const marker2: Position = { x: originPos.x + 2, y: originPos.y };

        gameState.board.markers.set(positionToString(marker1), {
          player: 2,
          position: marker1,
          type: 'regular'
        });
        gameState.board.markers.set(positionToString(marker2), {
          player: 2,
          position: marker2,
          type: 'regular'
        });
      } else if (scenario.ref.id === 'Rules_8_2_Q2_marker_landing_own_vs_opponent_square8') {
        // Place one same-colour marker and one opponent marker at equal
        // minimum-distance radii so we can assert that landing on own
        // marker is allowed but landing on opponent marker is not.
        //
        // origin = (3,3); own marker at (5,3) (distance 2 east);
        // opponent marker at (3,5) (distance 2 south).
        if (scenario.boardType !== 'square8') {
          throw new Error('Rules_8_2_Q2_marker_landing_own_vs_opponent_square8 must use square8');
        }
        const originPos: Position =
          scenario.origin.z != null
            ? pos(scenario.origin.x, scenario.origin.y, scenario.origin.z)
            : pos(scenario.origin.x, scenario.origin.y);

        const ownMarker: Position = { x: originPos.x + 2, y: originPos.y };
        const oppMarker: Position = { x: originPos.x, y: originPos.y + 2 };

        gameState.board.markers.set(positionToString(ownMarker), {
          player: 1,
          position: ownMarker,
          type: 'regular'
        });
        gameState.board.markers.set(positionToString(oppMarker), {
          player: 2,
          position: oppMarker,
          type: 'regular'
        });
      }
 
      const moves = ruleEngine.getValidMoves(gameState);
      const movementMoves = moves.filter(
        (m) => m.type === 'move_stack' || m.type === 'move_ring'
      ) as Move[];
 
      if (scenario.ref.id.startsWith('Rules_8_2_Q2_minimum_distance')) {
        // Minimum distance invariant: there must be at least one legal move with
        // distance >= stackHeight, and no move that is strictly shorter than
        // stackHeight.
        expect(movementMoves.length).toBeGreaterThan(0);
 
        const hasTooShortMove = movementMoves.some((m) => {
          if (!m.to || !m.from) return false;
 
          if (scenario.boardType === 'hexagonal') {
            const dx = (m.to.x || 0) - (m.from.x || 0);
            const dy = (m.to.y || 0) - (m.from.y || 0);
            const dz = (m.to.z || 0) - (m.from.z || 0);
            const dist = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
            return dist < scenario.stackHeight;
          }
 
          const dx = Math.abs(m.to.x - m.from.x);
          const dy = Math.abs(m.to.y - m.from.y);
          const dist = Math.max(dx, dy);
          return dist < scenario.stackHeight;
        });
 
        expect(hasTooShortMove).toBe(false);
      } else if (
        scenario.ref.id === 'Rules_8_2_Q2_markers_any_valid_space_beyond_square8'
      ) {
        // Marker-run invariant: for the square8 marker scenario we should be
        // able to land on any valid empty space beyond the markers on the
        // ray, not just the first such space, provided the minimum-distance
        // and path rules are satisfied.
        const originKey = positionToString(
          scenario.origin.z != null
            ? pos(scenario.origin.x, scenario.origin.y, scenario.origin.z)
            : pos(scenario.origin.x, scenario.origin.y)
        );
        const targetsFromOrigin = movementMoves
          .filter((m) => m.from && positionToString(m.from) === originKey)
          .map((m) => positionToString(m.to));
 
        const from = scenario.origin;
        const landing1 = positionToString({ x: from.x + 3, y: from.y });
        const landing2 = positionToString({ x: from.x + 4, y: from.y });
 
        expect(targetsFromOrigin).toContain(landing1);
        expect(targetsFromOrigin).toContain(landing2);
      } else if (
        scenario.ref.id === 'Rules_8_2_Q2_marker_landing_own_vs_opponent_square8' ||
        scenario.ref.id === 'Rules_8_2_Q2_marker_landing_own_vs_opponent_hexagonal'
      ) {
        // Marker-landing invariant: with one same-colour marker and one
        // opponent marker at the same minimum-distance radius, landing on the
        // own marker must be allowed but landing on the opponent marker must
        // be disallowed. This is tested on both square8 and hex boards using
        // appropriate distance metrics.
        const originKey = positionToString(
          scenario.origin.z != null
            ? pos(scenario.origin.x, scenario.origin.y, scenario.origin.z)
            : pos(scenario.origin.x, scenario.origin.y)
        );
        const targetsFromOrigin = movementMoves
          .filter((m) => m.from && positionToString(m.from) === originKey)
          .map((m) => positionToString(m.to));
 
        const from = scenario.origin;
 
        if (scenario.boardType === 'square8') {
          const ownKey = positionToString({ x: from.x + 2, y: from.y });
          const oppKey = positionToString({ x: from.x, y: from.y + 2 });
 
          expect(targetsFromOrigin).toContain(ownKey);
          expect(targetsFromOrigin).not.toContain(oppKey);
        } else if (scenario.boardType === 'hexagonal') {
          const ownKey = positionToString({
            x: from.x + 2,
            y: from.y - 2,
            z: from.z != null ? from.z : 0
          });
          const oppKey = positionToString({
            x: from.x - 2,
            y: from.y + 2,
            z: from.z != null ? from.z : 0
          });
 
          expect(targetsFromOrigin).toContain(ownKey);
          expect(targetsFromOrigin).not.toContain(oppKey);
        } else {
          throw new Error(
            `Unsupported boardType for marker landing scenario: ${scenario.boardType as string}`
          );
        }
      } else if (
        scenario.ref.id === 'Rules_8_3_Q3_blocked_by_stacks_and_collapsed_square8'
      ) {
        // Blocking invariant: no legal move may "jump over" the blocking stack
        // or collapsed space along their rays.
        const stackBlock = blockers!.find((b) => b.type === 'stack')!;
        const collapsedBlock = blockers!.find((b) => b.type === 'collapsed')!;
 
        const illegalThroughStack = movementMoves.some((m) => {
          if (!m.to || !m.from) return false;
          return (
            m.from.x === origin.x &&
            m.from.y === origin.y &&
            m.to.y === origin.y &&
            m.to.x > (stackBlock.position.x ?? 0)
          );
        });
 
        const illegalThroughCollapsed = movementMoves.some((m) => {
          if (!m.to || !m.from) return false;
          return (
            m.from.x === origin.x &&
            m.from.y === origin.y &&
            m.to.x === origin.x &&
            m.to.y > (collapsedBlock.position.y ?? 0)
          );
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
