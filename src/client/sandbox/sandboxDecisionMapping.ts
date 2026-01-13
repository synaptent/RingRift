/**
 * @fileoverview Sandbox Decision Mapping - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This module is an **adapter** between the canonical orchestrator
 * and the client UI. It does not contain rules logic.
 *
 * Canonical SSoT:
 * - Decision types: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - PlayerChoice types: `src/shared/types/game.ts`
 *
 * This adapter:
 * - Maps PendingDecision (orchestrator format) → PlayerChoice (UI format)
 * - Maps PlayerChoiceResponse (UI) → Move (engine format)
 * - Pure, stateless functions with no rules logic
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import type { BoardState, Move, PendingDecision, Position } from '../../shared/engine';
import { positionToString, isCaptureMove } from '../../shared/engine';
import type {
  PlayerChoice,
  PlayerChoiceResponse,
  RingEliminationChoice,
  LineOrderChoice,
  LineRewardChoice,
  RegionOrderChoice,
  CaptureDirectionChoice,
} from '../../shared/types/game';

/**
 * Context needed for decision mapping - minimal subset of GameState.
 */
export interface DecisionMappingContext {
  gameId: string;
  board: BoardState;
}

/**
 * Generate a unique ID for a player choice.
 */
function generateChoiceId(prefix: string): string {
  return `sandbox-${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/**
 * Map a PendingDecision from the orchestrator to a PlayerChoice for the UI.
 *
 * This is a pure function that converts orchestrator decision format to the
 * UI-facing PlayerChoice format used by the interaction handler.
 */
export function mapPendingDecisionToPlayerChoice(
  decision: PendingDecision,
  context: DecisionMappingContext
): PlayerChoice {
  const decisionType = decision.type;
  const options = decision.options;

  switch (decisionType) {
    case 'elimination_target': {
      const eliminationMoves = options.filter(
        (opt: Move) => opt.type === 'eliminate_rings_from_stack' && opt.to
      );

      // Derive eliminationContext from first move (all moves in a decision share same context)
      const firstMove = eliminationMoves[0];
      const eliminationContext = firstMove?.eliminationContext || 'territory';

      // Generate context-specific prompt
      let eliminationPrompt: string;
      if (decision.context?.description) {
        eliminationPrompt = decision.context.description;
      } else if (eliminationContext === 'line') {
        eliminationPrompt =
          'Line reward cost: You must eliminate ONE ring from the top of any stack you control.';
      } else if (eliminationContext === 'forced') {
        eliminationPrompt =
          'Forced elimination: You must eliminate your ENTIRE CAP from a controlled stack.';
      } else {
        eliminationPrompt =
          'Territory cost: You must eliminate your ENTIRE CAP from an eligible stack outside the region.';
      }

      const choice: RingEliminationChoice = {
        id: generateChoiceId('ring-elimination'),
        gameId: context.gameId,
        playerNumber: decision.player,
        type: 'ring_elimination',
        eliminationContext,
        prompt: eliminationPrompt,
        options: eliminationMoves.map((opt: Move, idx: number) => {
          const pos = opt.to as Position;
          const key = positionToString(pos);
          const stack = context.board.stacks.get(key);

          const capHeight =
            (opt.eliminationFromStack && opt.eliminationFromStack.capHeight) ||
            (stack ? stack.capHeight : 1);
          const totalHeight =
            (opt.eliminationFromStack && opt.eliminationFromStack.totalHeight) ||
            (stack ? stack.stackHeight : capHeight || 1);

          // Per RR-CANON-R122: line costs 1 ring; territory/forced costs entire cap
          const ringsToEliminate = eliminationContext === 'line' ? 1 : capHeight;

          return {
            stackPosition: pos,
            capHeight,
            totalHeight,
            ringsToEliminate,
            moveId: opt.id || `move-${idx}`,
          };
        }),
      };
      return choice;
    }

    case 'line_order': {
      const choice: LineOrderChoice = {
        id: generateChoiceId('line-order'),
        gameId: context.gameId,
        playerNumber: decision.player,
        type: 'line_order',
        prompt: 'Select which line to process first',
        options: options.map((opt: Move, idx: number) => ({
          lineId: `line-${idx}`,
          markerPositions: opt.formedLines?.[0]?.positions ?? [],
          moveId: opt.id || `move-${idx}`,
        })),
      };
      return choice;
    }

    case 'line_reward': {
      const choice: LineRewardChoice = {
        id: generateChoiceId('line-reward'),
        gameId: context.gameId,
        playerNumber: decision.player,
        type: 'line_reward_option',
        prompt: 'Choose reward for overlength line',
        options: options.map((opt: Move) => {
          const isCollapseAll =
            opt.collapsedMarkers?.length === opt.formedLines?.[0]?.positions?.length;
          return isCollapseAll
            ? ('option_1_collapse_all_and_eliminate' as const)
            : ('option_2_min_collapse_no_elimination' as const);
        }),
        moveIds: options.reduce(
          (acc, opt: Move, idx: number) => {
            const isCollapseAll =
              opt.collapsedMarkers?.length === opt.formedLines?.[0]?.positions?.length;
            const key = isCollapseAll
              ? 'option_1_collapse_all_and_eliminate'
              : 'option_2_min_collapse_no_elimination';
            acc[key] = opt.id || `move-${idx}`;
            return acc;
          },
          {} as Record<string, string>
        ),
      };
      return choice;
    }

    case 'region_order': {
      const choice: RegionOrderChoice = {
        id: generateChoiceId('region-order'),
        gameId: context.gameId,
        playerNumber: decision.player,
        type: 'region_order',
        prompt: 'Select which region to process first',
        options: options.map((opt: Move, idx: number) => {
          if (opt.type === 'skip_territory_processing') {
            return {
              regionId: 'skip',
              size: 0,
              representativePosition: { x: 0, y: 0 },
              moveId: opt.id || `move-${idx}`,
            };
          }

          const region = opt.disconnectedRegions?.[0];
          const representative = region?.spaces?.[0];
          // RR-FIX-2026-01-13: Use geometry-based stable regionId instead of array index.
          const stableRegionId = representative
            ? `region-${representative.x}-${representative.y}-${region?.spaces?.length ?? 0}`
            : `region-fallback-${idx}`;
          return {
            regionId: stableRegionId,
            size: region?.spaces?.length ?? 0,
            representativePosition: representative ?? { x: 0, y: 0 },
            moveId: opt.id || `move-${idx}`,
            spaces: region?.spaces,
          };
        }),
      };
      return choice;
    }

    case 'capture_direction': {
      // Filter to only capture moves for type-safe access to captureTarget
      const captureMoves = options.filter(isCaptureMove);
      const choice: CaptureDirectionChoice = {
        id: generateChoiceId('capture'),
        gameId: context.gameId,
        playerNumber: decision.player,
        type: 'capture_direction',
        prompt: 'Select capture direction',
        options: captureMoves.map((opt) => ({
          targetPosition: opt.captureTarget,
          landingPosition: opt.to,
          capturedCapHeight:
            context.board.stacks.get(positionToString(opt.captureTarget))?.capHeight ?? 0,
        })),
      };
      return choice;
    }

    case 'line_elimination_required': {
      // Line elimination after choosing the "collapse all + eliminate" reward
      // Maps to ring_elimination choice with 'line' context
      const eliminationMoves = options.filter(
        (opt: Move) => opt.type === 'eliminate_rings_from_stack' && opt.to
      );

      const lineEliminationChoice: RingEliminationChoice = {
        id: generateChoiceId('line-elimination'),
        gameId: context.gameId,
        playerNumber: decision.player,
        type: 'ring_elimination',
        eliminationContext: 'line',
        prompt:
          'Line reward cost: You must eliminate ONE ring from the top of any stack you control.',
        options: eliminationMoves.map((opt: Move, idx: number) => {
          const pos = opt.to as Position;
          const key = positionToString(pos);
          const stack = context.board.stacks.get(key);

          const capHeight =
            (opt.eliminationFromStack && opt.eliminationFromStack.capHeight) ||
            (stack ? stack.capHeight : 1);
          const totalHeight =
            (opt.eliminationFromStack && opt.eliminationFromStack.totalHeight) ||
            (stack ? stack.stackHeight : capHeight || 1);

          // Line elimination always costs 1 ring per RR-CANON-R122
          const ringsToEliminate = 1;

          return {
            stackPosition: pos,
            capHeight,
            totalHeight,
            ringsToEliminate,
            moveId: opt.id || `move-${idx}`,
          };
        }),
      };
      return lineEliminationChoice;
    }

    case 'chain_capture': {
      // Chain capture decision - player must choose next capture direction
      // Maps to capture_direction choice (same as initial capture_direction)
      const chainCaptureMoves = options.filter(isCaptureMove);
      const chainCaptureChoice: CaptureDirectionChoice = {
        id: generateChoiceId('chain-capture'),
        gameId: context.gameId,
        playerNumber: decision.player,
        type: 'capture_direction',
        prompt: 'Chain capture: Select next capture direction',
        options: chainCaptureMoves.map((opt) => ({
          targetPosition: opt.captureTarget,
          landingPosition: opt.to,
          capturedCapHeight:
            context.board.stacks.get(positionToString(opt.captureTarget))?.capHeight ?? 0,
        })),
      };
      return chainCaptureChoice;
    }

    default:
      // Generic fallback - use line_order as default
      return {
        id: generateChoiceId('decision'),
        gameId: context.gameId,
        playerNumber: decision.player,
        type: 'line_order',
        prompt: String(decision.type),
        options: [],
      };
  }
}

/**
 * Extended response type for mapping back to moves.
 */
export interface ExtendedPlayerChoiceResponse extends PlayerChoiceResponse<unknown> {
  selectedLineIndex?: number;
  selectedRegionIndex?: number;
}

/**
 * Map a PlayerChoice response back to the corresponding Move.
 *
 * This function finds the matching move from the original decision options
 * based on the user's selection.
 */
export function mapPlayerChoiceResponseToMove(
  decision: PendingDecision,
  response: ExtendedPlayerChoiceResponse
): Move {
  const options = decision.options;

  // Try to match based on response content
  if (response.selectedLineIndex !== undefined && options[response.selectedLineIndex]) {
    return options[response.selectedLineIndex];
  }

  if (response.selectedRegionIndex !== undefined && options[response.selectedRegionIndex]) {
    return options[response.selectedRegionIndex];
  }

  if (response.selectedOption) {
    // Type-safe extraction of selected option properties
    const selectedOpt = response.selectedOption as {
      moveId?: string;
      stackPosition?: Position;
      targetPosition?: Position;
      landingPosition?: Position;
    };

    // Ring elimination: prefer explicit moveId mapping when available,
    // falling back to matching by stack position.
    if (response.choiceType === 'ring_elimination' && selectedOpt.moveId) {
      const byId = options.find((opt: Move) => opt.id === selectedOpt.moveId);
      if (byId) {
        return byId;
      }

      const selectedPos = selectedOpt.stackPosition;
      if (selectedPos) {
        const selectedKey = positionToString(selectedPos);
        const byPos = options.find((opt: Move) => {
          if (!opt.to) return false;
          return positionToString(opt.to as Position) === selectedKey;
        });
        if (byPos) {
          return byPos;
        }
      }
    }

    // RR-FIX-2026-01-13: Region order - use moveId matching for stable region identification.
    // This avoids relying on index-based regionIds which become invalid after regions are processed.
    if (response.choiceType === 'region_order' && selectedOpt.moveId) {
      const byId = options.find((opt: Move) => opt.id === selectedOpt.moveId);
      if (byId) {
        return byId;
      }
    }

    // Match by position for capture direction
    const selected = options.filter(isCaptureMove).find((opt) => {
      if (selectedOpt.targetPosition && selectedOpt.landingPosition) {
        return (
          positionToString(opt.captureTarget) === positionToString(selectedOpt.targetPosition) &&
          positionToString(opt.to) === positionToString(selectedOpt.landingPosition)
        );
      }
      return false;
    });
    if (selected) return selected;
  }

  // Default to first option
  return options[0];
}

/**
 * Build a capture direction choice for chain capture prompts.
 */
export function buildCaptureDirectionChoice(
  gameId: string,
  playerNumber: number,
  captureMoves: Move[],
  board: BoardState
): CaptureDirectionChoice {
  const typedCaptures = captureMoves.filter(isCaptureMove);

  return {
    id: generateChoiceId('capture'),
    gameId,
    playerNumber,
    type: 'capture_direction',
    prompt: 'Select capture direction',
    options: typedCaptures.map((move) => ({
      targetPosition: move.captureTarget,
      landingPosition: move.to,
      capturedCapHeight: board.stacks.get(positionToString(move.captureTarget))?.capHeight ?? 0,
    })),
  };
}

/**
 * Build a region order choice for territory processing prompts.
 */
export function buildRegionOrderChoice(
  gameId: string,
  playerNumber: number,
  regionMoves: Move[],
  prompt: string = 'Territory claimed – choose area to process'
): RegionOrderChoice {
  return {
    id: generateChoiceId('region'),
    gameId,
    playerNumber,
    type: 'region_order',
    prompt,
    options: regionMoves.map((opt, index) => {
      const region = opt.disconnectedRegions?.[0];
      const representative = region?.spaces?.[0];
      // RR-FIX-2026-01-13: Use geometry-based stable regionId instead of array index.
      const stableRegionId = representative
        ? `region-${representative.x}-${representative.y}-${region?.spaces?.length ?? 0}`
        : `region-fallback-${index}`;
      const regionKey = representative
        ? `${representative.x},${representative.y}`
        : `region-${index}`;

      return {
        regionId: stableRegionId,
        size: region?.spaces?.length ?? 0,
        representativePosition: representative ?? { x: 0, y: 0 },
        moveId: opt.id || `process-region-${stableRegionId}-${regionKey}`,
        spaces: region?.spaces,
      };
    }),
  };
}

/**
 * Build a ring elimination choice for territory self-elimination prompts.
 */
export function buildRingEliminationChoice(
  gameId: string,
  playerNumber: number,
  eliminationMoves: Move[],
  board: BoardState,
  prompt: string
): RingEliminationChoice {
  return {
    id: generateChoiceId('territory-elim'),
    gameId,
    playerNumber,
    type: 'ring_elimination',
    prompt,
    options: eliminationMoves.map((opt: Move) => {
      const pos = opt.to as Position;
      const key = positionToString(pos);
      const stack = board.stacks.get(key);

      const capHeight =
        (opt.eliminationFromStack && opt.eliminationFromStack.capHeight) ||
        (stack ? stack.capHeight : 1);
      const totalHeight =
        (opt.eliminationFromStack && opt.eliminationFromStack.totalHeight) ||
        (stack ? stack.stackHeight : capHeight || 1);

      return {
        stackPosition: pos,
        capHeight,
        totalHeight,
        // For territory eliminations, ringsToEliminate equals capHeight (RR-CANON-R145)
        ringsToEliminate: capHeight,
        moveId: opt.id || key,
      };
    }),
  };
}
