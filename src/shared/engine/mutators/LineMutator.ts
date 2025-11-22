import { GameState, ProcessLineAction, ChooseLineRewardAction } from '../types';
import { positionToString, BOARD_CONFIGS, Position } from '../../types/game';

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
  const config = BOARD_CONFIGS[state.board.type];

  if (line.length > config.lineLength) {
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
      // All rings in stack are eliminated
      // "Any rings on these spaces are returned to their owners' supplies (eliminated)"
      // Wait, "returned to supply" usually means back to hand?
      // Rule 12.1: "Any rings occupying these spaces are removed from the board and returned to their owners."
      // "Returned to owners" usually implies back to hand (available to be placed again).
      // BUT, "Eliminated" means removed from game.
      // Let's check definitions.
      // Rule 5.3: "Rings removed from the board are returned to the owning player's supply unless specified as 'eliminated'."
      // Rule 12.1 (Line Completion): "Option 1: Collapse the entire line... All spaces... become collapsed territory... Any rings... are returned to their owners."
      // It does NOT say "eliminated".
      // So they go back to hand.

      for (const ringOwner of stack.rings) {
        const p = newState.players.find((pl) => pl.playerNumber === ringOwner);
        if (p) {
          p.ringsInHand++;
          newState.totalRingsInPlay--;
        }
      }
      newState.board.stacks.delete(key);
    }

    // Remove marker if any
    if (newState.board.markers.has(key)) {
      newState.board.markers.delete(key);
    }

    // Mark as collapsed territory
    newState.board.collapsedSpaces.set(key, newState.currentPlayer);
  }

  // 2. Handle Elimination Reward (Option 1 only)
  // If we collapsed MORE than the minimum (i.e. the whole line, and it was > min), we get to eliminate a ring?
  // Rule 12.1 Option 1: "Collapse the entire line... In addition, you may choose one of your opponent's rings... and eliminate it."
  // Wait, is this ALWAYS for Option 1?
  // "Option 1: Collapse the entire line... (Reward: Territory + Potential Ring Elimination)"
  // "If the line is longer than the minimum... you may eliminate one opponent ring."
  // If it is EXACTLY the minimum, no elimination reward?
  // Rule 12.1: "If the line is exactly [L] spaces long... collapse the line... (No ring elimination reward)."

  // So:
  // - If positionsToCollapse.length > minLength -> Trigger Elimination Choice?
  // - Or is it automatic?
  // The rule says "you may choose". This implies a follow-up action/phase.

  // However, for this subtask, we are implementing the MUTATOR.
  // If an elimination is required, we might need to transition to a state where that happens.
  // OR, the action payload should include the elimination target?
  // The `ChooseLineRewardAction` doesn't have an elimination target field.
  // `ProcessLineAction` doesn't either.

  // Let's check `GameAction` types again.
  // `EliminateStackAction` exists!
  // So, if an elimination is earned, we should probably transition to a state that allows `ELIMINATE_STACK`.

  // But wait, `ELIMINATE_STACK` is noted as "// For Forced Elimination Choice" in types.ts.
  // Is it also used for Line Reward Elimination?
  // The context says: "Gap Closure: Implement the ELIMINATE_STACK action in the TerritoryMutator... to handle the 'Forced Elimination Choice'".

  // What about Line Reward?
  // Maybe we just handle the collapse here, and if a reward is earned, the GameEngine (orchestrator)
  // will detect that and set the next phase/state to allow elimination?

  // For now, let's just handle the collapse.

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
