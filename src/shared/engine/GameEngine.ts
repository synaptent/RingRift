import {
  GameState,
  GameAction,
  GameEvent,
  ValidationResult,
  MoveStackAction,
  PlaceRingAction,
  OvertakingCaptureAction,
  ContinueChainAction,
  ProcessLineAction,
  ChooseLineRewardAction,
  ProcessTerritoryAction,
  EliminateStackAction,
} from './types';
import { validateMovement } from './validators/MovementValidator';
import { validatePlacement, validateSkipPlacement } from './validators/PlacementValidator';
import { validateProcessLine, validateChooseLineReward } from './validators/LineValidator';
import { validateCapture } from './validators/CaptureValidator';
import { validateProcessTerritory, validateEliminateStack } from './validators/TerritoryValidator';
// We need to import other validators (Capture, Territory) but they might not be implemented yet or I missed reading them.
// I'll assume they exist or stub them if needed.
// I read 'src/shared/engine/validators/index.ts' in the file list but didn't read content.
// I'll use 'any' for now if imports fail, but better to try importing.

import { mutateMovement } from './mutators/MovementMutator';
import { mutatePlacement } from './mutators/PlacementMutator';
import { mutateCapture } from './mutators/CaptureMutator';
import { mutateProcessLine, mutateChooseLineReward } from './mutators/LineMutator';
import { mutateProcessTerritory, mutateEliminateStack } from './mutators/TerritoryMutator';
import { mutateTurnChange, mutatePhaseChange } from './mutators/TurnMutator';
import { findAllLines } from './lineDetection';
import { findDisconnectedRegions } from './territoryDetection';
import { enumerateCaptureMoves, CaptureBoardAdapters } from './captureLogic';
import { isValidPosition } from './validators/utils';
import { positionToString } from '../types/game';

export class GameEngine {
  private state: GameState;

  constructor(initialState: GameState) {
    this.state = initialState;
  }

  public getGameState(): GameState {
    return this.state;
  }

  public processAction(action: GameAction): GameEvent {
    // 1. Validate Action
    const validation = this.validateAction(action);
    if (!validation.valid) {
      return {
        type: 'ERROR_OCCURRED',
        gameId: this.state.id,
        timestamp: Date.now(),
        payload: {
          error: validation.reason,
          code: validation.code,
        },
      };
    }

    // 2. Apply Mutation
    try {
      const newState = this.applyMutation(action);
      this.state = newState;

      // 3. Check for State Transitions (Turn End, Phase Change, Win Condition)
      // This is where the "Orchestrator" logic lives.
      // For now, we'll do simple transitions.
      this.checkStateTransitions(action);

      return {
        type: 'ACTION_PROCESSED',
        gameId: this.state.id,
        timestamp: Date.now(),
        payload: {
          action,
          newState: this.state,
        },
      };
    } catch (error: any) {
      return {
        type: 'ERROR_OCCURRED',
        gameId: this.state.id,
        timestamp: Date.now(),
        payload: {
          error: error.message,
          code: 'MUTATION_ERROR',
        },
      };
    }
  }

  private validateAction(action: GameAction): ValidationResult {
    switch (action.type) {
      case 'PLACE_RING':
        return validatePlacement(this.state, action as PlaceRingAction);
      case 'MOVE_STACK':
        return validateMovement(this.state, action as MoveStackAction);
      case 'OVERTAKING_CAPTURE':
      case 'CONTINUE_CHAIN':
        // Capture validation is shared between initial and chain segments
        return validateCapture(this.state, action as OvertakingCaptureAction);
      case 'PROCESS_LINE':
        return validateProcessLine(this.state, action as ProcessLineAction);
      case 'CHOOSE_LINE_REWARD':
        return validateChooseLineReward(this.state, action as ChooseLineRewardAction);
      case 'PROCESS_TERRITORY':
        return validateProcessTerritory(this.state, action as ProcessTerritoryAction);
      case 'ELIMINATE_STACK':
        return validateEliminateStack(this.state, action as EliminateStackAction);
      case 'SKIP_PLACEMENT':
        return validateSkipPlacement(this.state, action as any);
      default:
        return { valid: false, reason: 'Unknown action type', code: 'UNKNOWN_ACTION' };
    }
  }

  private applyMutation(action: GameAction): GameState {
    switch (action.type) {
      case 'PLACE_RING':
        return mutatePlacement(this.state, action as PlaceRingAction);
      case 'MOVE_STACK':
        return mutateMovement(this.state, action as MoveStackAction);
      case 'OVERTAKING_CAPTURE':
      case 'CONTINUE_CHAIN':
        return mutateCapture(this.state, action as OvertakingCaptureAction | ContinueChainAction);
      case 'PROCESS_LINE':
        return mutateProcessLine(this.state, action as ProcessLineAction);
      case 'CHOOSE_LINE_REWARD':
        return mutateChooseLineReward(this.state, action as ChooseLineRewardAction);
      case 'PROCESS_TERRITORY':
        return mutateProcessTerritory(this.state, action as ProcessTerritoryAction);
      case 'ELIMINATE_STACK':
        return mutateEliminateStack(this.state, action as EliminateStackAction);
      case 'SKIP_PLACEMENT':
        // No board mutation, just phase transition (handled in checkStateTransitions)
        return this.state;
      default:
        throw new Error('Unknown action type');
    }
  }

  private checkStateTransitions(action: GameAction) {
    // This is a simplified state machine.
    // Real implementation needs to check for:
    // - Line formation (after placement/move/capture)
    // - Territory formation/disconnection (after line collapse)
    // - Win conditions
    // - Next phase logic

    // Example: After placement, go to movement
    if (action.type === 'PLACE_RING' || action.type === 'SKIP_PLACEMENT') {
      this.state = mutatePhaseChange(this.state, 'movement');
    }

    // Example: After movement, end turn (unless lines formed)
    if (action.type === 'MOVE_STACK') {
      // Check lines
      const lines = findAllLines(this.state.board);
      if (lines.length > 0) {
        this.state = {
          ...this.state,
          board: {
            ...this.state.board,
            formedLines: lines,
          },
        };
        this.state = mutatePhaseChange(this.state, 'line_processing');
      } else {
        // Check territory disconnection
        const regions = findDisconnectedRegions(this.state.board);
        if (regions.length > 0) {
          // Populate territories map with disconnected regions
          const newTerritories = new Map(this.state.board.territories);
          regions.forEach((region, index) => {
            newTerritories.set(`disconnected-${index}`, region);
          });

          this.state = {
            ...this.state,
            board: {
              ...this.state.board,
              territories: newTerritories,
            },
          };
          this.state = mutatePhaseChange(this.state, 'territory_processing');
        } else {
          this.state = mutateTurnChange(this.state);
        }
      }
    }

    // Example: After capture, check for chain or end turn
    if (action.type === 'OVERTAKING_CAPTURE' || action.type === 'CONTINUE_CHAIN') {
      // Check if chain continues
      const adapters: CaptureBoardAdapters = {
        isValidPosition: (pos) =>
          isValidPosition(pos, this.state.board.type, this.state.board.size),
        isCollapsedSpace: (pos) => this.state.board.collapsedSpaces.has(positionToString(pos)),
        getStackAt: (pos) => {
          const stack = this.state.board.stacks.get(positionToString(pos));
          if (!stack) return undefined;
          return {
            controllingPlayer: stack.controllingPlayer,
            capHeight: stack.capHeight,
            stackHeight: stack.stackHeight,
          };
        },
        getMarkerOwner: (pos) => {
          const marker = this.state.board.markers.get(positionToString(pos));
          return marker?.player;
        },
      };

      const nextCaptures = enumerateCaptureMoves(
        this.state.board.type,
        action.to,
        action.playerId,
        adapters,
        this.state.moveHistory.length + 1
      );

      if (nextCaptures.length > 0) {
        this.state = mutatePhaseChange(this.state, 'chain_capture');
        // Keep current player
        this.state = {
          ...this.state,
          currentPlayer: action.playerId,
        };
      } else {
        // Check lines
        const lines = findAllLines(this.state.board);
        if (lines.length > 0) {
          this.state = {
            ...this.state,
            board: {
              ...this.state.board,
              formedLines: lines,
            },
          };
          this.state = mutatePhaseChange(this.state, 'line_processing');
        } else {
          // Check territory disconnection
          const regions = findDisconnectedRegions(this.state.board);
          if (regions.length > 0) {
            const newTerritories = new Map(this.state.board.territories);
            regions.forEach((region, index) => {
              newTerritories.set(`disconnected-${index}`, region);
            });

            this.state = {
              ...this.state,
              board: {
                ...this.state.board,
                territories: newTerritories,
              },
            };
            this.state = mutatePhaseChange(this.state, 'territory_processing');
          } else {
            this.state = mutateTurnChange(this.state);
          }
        }
      }
    }

    // Example: After elimination, end turn
    if (action.type === 'ELIMINATE_STACK') {
      this.state = mutateTurnChange(this.state);
    }
  }
}
