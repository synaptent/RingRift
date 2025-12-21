import { Move, positionToString } from '../types/game';
import type { Position, Territory, LineInfo } from '../types/game';
import { flagEnabled, debugLog } from '../utils/envFlags';
import { getRingsToEliminate, type EliminationContext } from './aggregates/EliminationAggregate';
import { isLegacyMoveType } from './legacy/legacyMoveTypes';
import {
  GameState as EngineGameState,
  GameAction,
  PlaceRingAction,
  MoveStackAction,
  OvertakingCaptureAction,
  ContinueChainAction,
  ProcessLineAction,
  ChooseLineRewardAction,
  ProcessTerritoryAction,
  EliminateStackAction,
  SkipPlacementAction,
} from './types';

/**
 * Adapter between the canonical wire-level {@link Move} representation used
 * by the backend GameEngine / transports and the internal {@link GameAction}
 * union used by the shared engine.
 *
 * The mapping is intentionally lossy: diagnostic fields on {@link Move}
 * (stackMoved, capturedStacks, captureChain, etc.) are ignored when
 * constructing {@link GameAction} values.
 */
export class MoveMappingError extends Error {
  constructor(
    message: string,
    public readonly move?: Move
  ) {
    super(message);
    this.name = 'MoveMappingError';
  }
}

/**
 * Convert a canonical {@link Move} into a shared-engine {@link GameAction}.
 *
 * The current engine {@link EngineGameState} is required for decision
 * phases that index into `board.formedLines` or `board.territories`.
 */
export function moveToGameAction(move: Move, state: EngineGameState): GameAction {
  if (isLegacyMoveType(move.type)) {
    throw new MoveMappingError(
      `Move type ${move.type} is legacy-only and is not supported by the shared-engine adapter`,
      move
    );
  }

  switch (move.type) {
    case 'place_ring':
      return mapPlaceRingMove(move);
    case 'skip_placement':
      return mapSkipPlacementMove(move);
    case 'move_stack':
      return mapMoveStackMove(move);
    case 'overtaking_capture':
      return mapOvertakingCaptureMove(move);
    case 'continue_capture_segment':
      return mapContinueCaptureMove(move);
    case 'process_line':
      return mapProcessLineMove(move, state);
    case 'choose_line_option':
      return mapChooseLineRewardMove(move, state);
    case 'choose_territory_option':
      return mapProcessTerritoryRegionMove(move, state);
    case 'eliminate_rings_from_stack':
      return mapEliminateRingsFromStackMove(move);
    default: {
      const exhaustive: never = move.type as never;
      throw new MoveMappingError(`Unknown Move type ${String(exhaustive) || move.type}`, move);
    }
  }
}

/**
 * Move payload used for round-tripping without id/timestamp/moveNumber.
 */
export type BareMove = Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;

/**
 * Convert a shared-engine {@link GameAction} back into a canonical
 * {@link Move} payload. The `before` state is used to recover
 * line/territory metadata for decision phases.
 */
export function gameActionToMove(action: GameAction, before: EngineGameState): BareMove {
  switch (action.type) {
    case 'PLACE_RING':
      return actionToPlaceRingMove(action as PlaceRingAction, before);
    case 'SKIP_PLACEMENT':
      return actionToSkipPlacementMove(action as SkipPlacementAction);
    case 'MOVE_STACK':
      return actionToMoveStackMove(action as MoveStackAction);
    case 'OVERTAKING_CAPTURE':
      return actionToOvertakingCaptureMove(action as OvertakingCaptureAction);
    case 'CONTINUE_CHAIN':
      return actionToContinueChainMove(action as ContinueChainAction);
    case 'PROCESS_LINE':
      return actionToProcessLineMove(action as ProcessLineAction, before);
    case 'CHOOSE_LINE_REWARD':
      return actionToChooseLineRewardMove(action as ChooseLineRewardAction, before);
    case 'PROCESS_TERRITORY':
      return actionToProcessTerritoryMove(action as ProcessTerritoryAction, before);
    case 'ELIMINATE_STACK':
      return actionToEliminateStackMove(action as EliminateStackAction, before);
    default: {
      const exhaustive: never = action;
      throw new MoveMappingError(
        `Unsupported GameAction type ${(exhaustive as { type?: string }).type ?? 'unknown'}`
      );
    }
  }
}

function mapPlaceRingMove(move: Move): PlaceRingAction {
  return {
    type: 'PLACE_RING',
    playerId: move.player,
    position: move.to,
    count: move.placementCount ?? 1,
  };
}

function mapSkipPlacementMove(move: Move): SkipPlacementAction {
  return {
    type: 'SKIP_PLACEMENT',
    playerId: move.player,
  };
}

function mapMoveStackMove(move: Move): MoveStackAction {
  if (!move.from) {
    throw new MoveMappingError('move_stack Move is missing from position', move);
  }
  return {
    type: 'MOVE_STACK',
    playerId: move.player,
    from: move.from,
    to: move.to,
  };
}

function mapOvertakingCaptureMove(move: Move): OvertakingCaptureAction {
  if (!move.from || !move.captureTarget) {
    throw new MoveMappingError('overtaking_capture Move is missing from or captureTarget', move);
  }
  return {
    type: 'OVERTAKING_CAPTURE',
    playerId: move.player,
    from: move.from,
    to: move.to,
    captureTarget: move.captureTarget,
  };
}

function mapContinueCaptureMove(move: Move): ContinueChainAction {
  if (!move.from || !move.captureTarget) {
    throw new MoveMappingError(
      'continue_capture_segment Move is missing from or captureTarget',
      move
    );
  }
  return {
    type: 'CONTINUE_CHAIN',
    playerId: move.player,
    from: move.from,
    to: move.to,
    captureTarget: move.captureTarget,
  };
}

function mapProcessLineMove(move: Move, state: EngineGameState): ProcessLineAction {
  const lineIndex = resolveLineIndexFromMove(move, state);
  return {
    type: 'PROCESS_LINE',
    playerId: move.player,
    lineIndex,
  };
}

function mapChooseLineRewardMove(move: Move, state: EngineGameState): ChooseLineRewardAction {
  const lineIndex = resolveLineIndexFromMove(move, state);
  const lines = state.board.formedLines as unknown as Array<{
    player: number;
    positions: Position[];
  }>;
  const line = lines[lineIndex];

  const collapsedPositions =
    move.collapsedMarkers && move.collapsedMarkers.length > 0 ? move.collapsedMarkers : undefined;

  let selection: ChooseLineRewardAction['selection'] = 'COLLAPSE_ALL';

  if (collapsedPositions) {
    // If collapsed positions match the full line length, it's COLLAPSE_ALL (explicitly provided).
    // Otherwise, it's MINIMUM_COLLAPSE.
    if (line && collapsedPositions.length === line.positions.length) {
      selection = 'COLLAPSE_ALL';
    } else {
      debugLog(
        flagEnabled('RINGRIFT_TRACE_DEBUG'),
        'DEBUG: mapChooseLineRewardMove defaulting to MINIMUM_COLLAPSE',
        {
          lineLength: line?.positions.length,
          collapsedLength: collapsedPositions.length,
          linePositions: line?.positions,
          collapsedPositions,
        }
      );
      selection = 'MINIMUM_COLLAPSE';
    }
  }

  const base: ChooseLineRewardAction = {
    type: 'CHOOSE_LINE_REWARD',
    playerId: move.player,
    lineIndex,
    selection,
  };
  return collapsedPositions ? { ...base, collapsedPositions } : base;
}

function mapProcessTerritoryRegionMove(move: Move, state: EngineGameState): ProcessTerritoryAction {
  const regionId = resolveRegionIdFromMove(move, state);
  return {
    type: 'PROCESS_TERRITORY',
    playerId: move.player,
    regionId,
  };
}

function mapEliminateRingsFromStackMove(move: Move): EliminateStackAction {
  if (!move.to) {
    throw new MoveMappingError('eliminate_rings_from_stack Move is missing stack position', move);
  }
  return {
    type: 'ELIMINATE_STACK',
    playerId: move.player,
    stackPosition: move.to,
    ...(move.eliminationContext ? { eliminationContext: move.eliminationContext } : {}),
  };
}

function resolveLineIndexFromMove(move: Move, state: EngineGameState): number {
  const lines = state.board.formedLines as unknown as Array<{
    player: number;
    positions: Position[];
  }>;
  if (!lines || lines.length === 0) {
    throw new MoveMappingError('No formedLines available on state for line-processing Move', move);
  }

  const candidate = move.formedLines && move.formedLines[0];
  if (candidate) {
    const targetKey = canonicalPositionsKey(candidate.positions);
    const index = lines.findIndex((line) => {
      if (line.player !== candidate.player) return false;
      return canonicalPositionsKey(line.positions) === targetKey;
    });
    if (index >= 0) {
      return index;
    }
    throw new MoveMappingError(
      'Could not match Move.formedLines[0] to any board.formedLines entry',
      move
    );
  }

  // Fallback: first line for this player. This mirrors legacy "first line wins"
  // behaviour when explicit metadata is unavailable.
  const fallbackIndex = lines.findIndex((line) => line.player === move.player);
  if (fallbackIndex >= 0) {
    return fallbackIndex;
  }
  throw new MoveMappingError('No line for Move.player found in board.formedLines', move);
}

function resolveRegionIdFromMove(move: Move, state: EngineGameState): string {
  const territories = state.board.territories as unknown as Map<string, Territory>;
  if (!territories || territories.size === 0) {
    throw new MoveMappingError(
      'No territories available on state for choose_territory_option Move',
      move
    );
  }

  const candidate = move.disconnectedRegions && move.disconnectedRegions[0];
  if (candidate) {
    const targetKeys = new Set(candidate.spaces.map((p) => positionToString(p)));
    for (const [id, region] of territories.entries()) {
      if (!region.isDisconnected) continue;
      if (region.controllingPlayer !== candidate.controllingPlayer) continue;
      const regionKeys = new Set(region.spaces.map((p) => positionToString(p)));
      if (regionKeys.size !== targetKeys.size) continue;
      let mismatch = false;
      for (const key of targetKeys) {
        if (!regionKeys.has(key)) {
          mismatch = true;
          break;
        }
      }
      if (!mismatch) {
        return id;
      }
    }
    debugLog(flagEnabled('RINGRIFT_TRACE_DEBUG'), 'DEBUG: resolveRegionIdFromMove failed', {
      candidateSpaces: Array.from(targetKeys),
      territories: Array.from(territories.entries()).map(([id, r]) => ({
        id,
        isDisconnected: r.isDisconnected,
        player: r.controllingPlayer,
        spaces: r.spaces.map((p) => positionToString(p)),
      })),
    });
    throw new MoveMappingError(
      'Could not match disconnectedRegions[0] to any board.territories entry',
      move
    );
  }

  // Fallback: first disconnected region controlled by the moving player.
  for (const [id, region] of territories.entries()) {
    if (region.isDisconnected && region.controllingPlayer === move.player) {
      return id;
    }
  }
  throw new MoveMappingError('No disconnected territory region found for moving player', move);
}

function canonicalPositionsKey(positions: Position[]): string {
  return positions.map((p) => positionToString(p)).join('|');
}

function actionToPlaceRingMove(action: PlaceRingAction, before: EngineGameState): BareMove {
  const key = positionToString(action.position);
  const existingStack = before.board.stacks.get(key);
  const placedOnStack = !!(existingStack && existingStack.rings.length > 0);
  return {
    type: 'place_ring',
    player: action.playerId,
    to: action.position,
    placementCount: action.count,
    placedOnStack,
    thinkTime: 0,
  };
}

function actionToSkipPlacementMove(_action: SkipPlacementAction): BareMove {
  return {
    type: 'skip_placement',
    player: _action.playerId,
    // Sentinel coordinate; has no semantic meaning for skip_placement.
    to: { x: 0, y: 0 },
    thinkTime: 0,
  };
}

function actionToMoveStackMove(action: MoveStackAction): BareMove {
  return {
    type: 'move_stack',
    player: action.playerId,
    from: action.from,
    to: action.to,
    thinkTime: 0,
  };
}

function actionToOvertakingCaptureMove(action: OvertakingCaptureAction): BareMove {
  return {
    type: 'overtaking_capture',
    player: action.playerId,
    from: action.from,
    to: action.to,
    captureTarget: action.captureTarget,
    thinkTime: 0,
  };
}

function actionToContinueChainMove(action: ContinueChainAction): BareMove {
  return {
    type: 'continue_capture_segment',
    player: action.playerId,
    from: action.from,
    to: action.to,
    captureTarget: action.captureTarget,
    thinkTime: 0,
  };
}

function actionToProcessLineMove(action: ProcessLineAction, before: EngineGameState): BareMove {
  const lines = before.board.formedLines as unknown as Array<{
    player: number;
    positions: Position[];
    length?: number;
    direction?: Position;
  }>;
  const line = lines[action.lineIndex];
  if (!line) {
    throw new MoveMappingError('PROCESS_LINE action references missing formedLines index');
  }
  const lineInfo: LineInfo = {
    player: line.player,
    positions: line.positions,
    length: line.length ?? line.positions.length,
    direction: line.direction ?? { x: 0, y: 0 },
  };
  return {
    type: 'process_line',
    player: action.playerId,
    // Use the first position in the line as a representative landing point.
    to: line.positions[0] ?? { x: 0, y: 0 },
    formedLines: [lineInfo],
    thinkTime: 0,
  } as BareMove;
}

function actionToChooseLineRewardMove(
  action: ChooseLineRewardAction,
  before: EngineGameState
): BareMove {
  const lines = before.board.formedLines as unknown as Array<{
    player: number;
    positions: Position[];
    length?: number;
    direction?: Position;
  }>;
  const line = lines[action.lineIndex];
  if (!line) {
    throw new MoveMappingError('CHOOSE_LINE_REWARD action references missing formedLines index');
  }
  const lineInfo: LineInfo = {
    player: line.player,
    positions: line.positions,
    length: line.length ?? line.positions.length,
    direction: line.direction ?? { x: 0, y: 0 },
  };
  const collapsedMarkers =
    action.selection === 'MINIMUM_COLLAPSE'
      ? (action.collapsedPositions ?? [])
      : // For COLLAPSE_ALL, embed the full line so the move is self-describing.
        line.positions;
  return {
    type: 'choose_line_option',
    player: action.playerId,
    to: line.positions[0] ?? { x: 0, y: 0 },
    formedLines: [lineInfo],
    thinkTime: 0,
    ...(collapsedMarkers && collapsedMarkers.length > 0 ? { collapsedMarkers } : {}),
  } as BareMove;
}

function actionToProcessTerritoryMove(
  action: ProcessTerritoryAction,
  before: EngineGameState
): BareMove {
  const territories = before.board.territories as unknown as Map<string, Territory>;
  const region = territories.get(action.regionId);
  if (!region) {
    throw new MoveMappingError('PROCESS_TERRITORY action references unknown regionId');
  }
  const representative = region.spaces[0] ?? { x: 0, y: 0 };
  return {
    type: 'choose_territory_option',
    player: action.playerId,
    to: representative,
    disconnectedRegions: [region],
    thinkTime: 0,
  } as BareMove;
}

function actionToEliminateStackMove(
  action: EliminateStackAction,
  before: EngineGameState
): BareMove {
  const key = positionToString(action.stackPosition);
  const stack = before.board.stacks.get(key);
  const capHeight = stack ? stack.capHeight : 0;
  const totalHeight = stack ? (stack.stackHeight ?? capHeight) : capHeight;
  const eliminationContext: EliminationContext = (action.eliminationContext ??
    'territory') as EliminationContext;
  let ringsToEliminate = 0;
  if (stack) {
    // Some adapter tests stub stacks without a `rings` array. Use canonical
    // elimination semantics when `rings` is present; otherwise, fall back to
    // the stack's capHeight metadata.
    if (Array.isArray(stack.rings)) {
      ringsToEliminate = getRingsToEliminate(stack, eliminationContext);
    } else if (eliminationContext === 'line' || eliminationContext === 'recovery') {
      ringsToEliminate = 1;
    } else {
      ringsToEliminate = capHeight;
    }
  }
  return {
    type: 'eliminate_rings_from_stack',
    player: action.playerId,
    to: action.stackPosition,
    thinkTime: 0,
    ...(action.eliminationContext ? { eliminationContext: action.eliminationContext } : {}),
    ...(ringsToEliminate > 0
      ? {
          eliminatedRings: [{ player: action.playerId, count: ringsToEliminate }],
          eliminationFromStack: {
            position: action.stackPosition,
            capHeight,
            totalHeight,
          },
        }
      : {}),
  } as BareMove;
}
