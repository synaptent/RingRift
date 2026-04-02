import { GameState, ProcessLineAction, ChooseLineRewardAction } from '../types';
import { positionToString, BoardType, Position } from '../../types/game';
import { getEffectiveLineLengthThreshold } from '../rulesConfig';

export function mutateProcessLine(state: GameState, action: ProcessLineAction): GameState {
  // If the line is exact length, we can process it immediately as Option 1 (Collapse All).
  // If it's longer, we technically need a choice.
  // However, the current architecture seems to imply `ProcessLineAction` might be used for the "automatic" case
  // or as a trigger.

  // For now, we will implement the logic:
  // If exact length -> Execute Option 1
  // If > exact length -> Throw error (Client should have sent ChooseLineRewardAction)
  // This enforces that ProcessLineAction is only for the "no choice needed" case.

  const line = state.board.formedLines[action.lineIndex];
  const requiredLength = getEffectiveLineLengthThreshold(
    state.board.type as BoardType,
    state.players.length
  );

  if (line.length > requiredLength) {
    throw new Error('LineMutator: Line length > minimum requires ChooseLineRewardAction');
  }

  // Execute Option 1: Collapse All
  return executeCollapse(state, line.positions, action.lineIndex);
}

export function mutateChooseLineReward(
  state: GameState,
  action: ChooseLineRewardAction
): GameState {
  const line = state.board.formedLines[action.lineIndex];

  if (action.selection === 'COLLAPSE_ALL') {
    return executeCollapse(state, line.positions, action.lineIndex);
  } else {
    // MINIMUM_COLLAPSE
    if (!action.collapsedPositions) {
      throw new Error('LineMutator: Missing collapsedPositions for MINIMUM_COLLAPSE');
    }
    return executeCollapse(state, action.collapsedPositions, action.lineIndex);
  }
}

function executeCollapse(
  state: GameState,
  positionsToCollapse: Position[],
  lineIndex: number
): GameState {
  const newState = {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
      markers: new Map(state.board.markers),
      collapsedSpaces: new Map(state.board.collapsedSpaces),
      formedLines: [...state.board.formedLines],
      eliminatedRings: { ...state.board.eliminatedRings },
    },
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
    totalRingsInPlay: number;
  };

  const player = newState.players.find((p) => p.playerNumber === newState.currentPlayer);
  if (!player) throw new Error('LineMutator: Player not found');

  // 1. Remove stacks/markers at collapsed positions and mark as collapsed
  for (const pos of positionsToCollapse) {
    const key = positionToString(pos);

    // Remove stack if any
    const stack = newState.board.stacks.get(key);
    if (stack) {
      // RR-CANON-R122 / Complete Rules Q7: any rings occupying newly collapsed
      // line spaces are permanently eliminated and credited to the acting player.
      const eliminatedCount = stack.rings.length;
      player.eliminatedRings += eliminatedCount;
      newState.board.eliminatedRings[newState.currentPlayer] =
        (newState.board.eliminatedRings[newState.currentPlayer] || 0) + eliminatedCount;
      newState.totalRingsEliminated = (newState.totalRingsEliminated || 0) + eliminatedCount;
      newState.totalRingsInPlay -= eliminatedCount;
      newState.board.stacks.delete(key);
    }

    // Remove marker if any
    if (newState.board.markers.has(key)) {
      newState.board.markers.delete(key);
    }

    // Mark as collapsed territory
    newState.board.collapsedSpaces.set(key, newState.currentPlayer);
  }

  // Update the player's territorySpaces to reflect newly collapsed spaces.
  // Per canonical rules, when a turn action causes collapse, that territory
  // is credited to the player taking the action.
  if (positionsToCollapse.length > 0) {
    player.territorySpaces += positionsToCollapse.length;
  }

  // 2. Line self-elimination cost / reward is handled as a separate
  // eliminate_rings_from_stack move by higher-level orchestrators.

  // 3. Remove the processed line from formedLines
  // We need to be careful about indices shifting if we remove.
  // But we are creating a new array.
  // Also, processing one line might break others?
  // "If multiple lines are formed... process them one by one."
  // "If processing one line breaks another... the broken line is discarded."

  // We remove the processed line.
  newState.board.formedLines.splice(lineIndex, 1);

  // We also need to check if other lines are broken by this collapse.
  // A line is broken if any of its positions are now collapsed.
  const collapsedKeys = new Set(positionsToCollapse.map((p) => positionToString(p)));

  newState.board.formedLines = newState.board.formedLines.filter((l) => {
    for (const pos of l.positions) {
      if (collapsedKeys.has(positionToString(pos))) {
        return false; // Line broken
      }
    }
    return true;
  });

  newState.lastMoveAt = new Date();
  return newState;
}
