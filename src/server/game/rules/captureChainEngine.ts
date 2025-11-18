import {
  GameState,
  Move,
  Position,
  positionToString,
  CaptureDirectionChoice,
  PlayerChoiceResponseFor
} from '../../../shared/types/game';
import { BoardManager } from '../BoardManager';
import { RuleEngine } from '../RuleEngine';
import { PlayerInteractionManager } from '../PlayerInteractionManager';

export interface ChainCaptureSegment {
  from: Position;
  target: Position;
  landing: Position;
  capturedCapHeight: number;
}

export interface ChainCaptureState {
  playerNumber: number;
  startPosition: Position;
  currentPosition: Position;
  segments: ChainCaptureSegment[];
  // Full capture moves (from=currentPosition) that the player may choose from
  availableMoves: Move[];
  // Positions visited by the capturing stack to help avoid pathological cycles
  visitedPositions: Set<string>;
}

export interface CaptureChainDeps {
  boardManager: BoardManager;
  ruleEngine: RuleEngine;
  interactionManager?: PlayerInteractionManager | undefined;
}

/**
 * Update or initialize the internal chain capture state after an
 * overtaking capture has been successfully applied to the board.
 *
 * This mirrors the original GameEngine.updateChainCaptureStateAfterCapture
 * implementation but is decoupled from the class.
 */
export function updateChainCaptureStateAfterCapture(
  state: ChainCaptureState | undefined,
  move: Move,
  capturedCapHeight: number
): ChainCaptureState | undefined {
  if (!move.from || !move.captureTarget || !move.to) {
    return state;
  }

  const segment: ChainCaptureSegment = {
    from: move.from,
    target: move.captureTarget,
    landing: move.to,
    capturedCapHeight
  };

  if (!state) {
    return {
      playerNumber: move.player,
      startPosition: move.from,
      currentPosition: move.to,
      segments: [segment],
      availableMoves: [],
      visitedPositions: new Set<string>([positionToString(move.from)])
    };
  }

  // Continuing an existing chain
  state.currentPosition = move.to;
  state.segments.push(segment);
  state.visitedPositions.add(positionToString(move.from));
  return state;
}

/**
 * Enumerate all valid capture moves from a given position for the
 * specified player by ray-walking in each movement direction and
 * validating each candidate via RuleEngine.
 *
 * This mirrors GameEngine.getCaptureOptionsFromPosition.
 */
export function getCaptureOptionsFromPosition(
  position: Position,
  playerNumber: number,
  gameState: GameState,
  deps: CaptureChainDeps
): Move[] {
  const { boardManager, ruleEngine } = deps;
  const board = gameState.board;
  const attackerStack = boardManager.getStack(position, board);

  if (!attackerStack || attackerStack.controllingPlayer !== playerNumber) {
    return [];
  }

  const moves: Move[] = [];
  const directions = boardManager.getConfig()
    ? getAllDirectionsForBoardType(gameState.boardType, boardManager)
    : [];

  for (const dir of directions) {
    // Step outward from the attacker to find the first potential target
    let step = 1;
    let targetPos: Position | undefined;

    for (;;) {
      const pos: Position = {
        x: position.x + dir.x * step,
        y: position.y + dir.y * step,
        ...(dir.z !== undefined && { z: (position.z || 0) + dir.z * step })
      };

      if (!boardManager.isValidPosition(pos)) {
        break; // Off-board
      }

      // Collapsed spaces block both target search and landing beyond
      if (boardManager.isCollapsedSpace(pos, board)) {
        break;
      }

      const stackAtPos = boardManager.getStack(pos, board);
      if (stackAtPos && stackAtPos.rings.length > 0) {
        // First stack encountered along this ray is the only possible
        // capture target in this direction.
        if (
          stackAtPos.controllingPlayer !== playerNumber &&
          attackerStack.capHeight >= stackAtPos.capHeight
        ) {
          targetPos = pos;
        }
        break;
      }

      step++;
    }

    if (!targetPos) continue;

    // From the target, walk further along the same ray to find candidate
    // landing positions. Each candidate is validated via validateMove to
    // ensure consistency with the RuleEngine's rules (distance, path,
    // landing legality, etc.).
    let landingStep = 1;
    for (;;) {
      const landingPos: Position = {
        x: targetPos.x + dir.x * landingStep,
        y: targetPos.y + dir.y * landingStep,
        ...(dir.z !== undefined && { z: (targetPos.z || 0) + dir.z * landingStep })
      };

      if (!boardManager.isValidPosition(landingPos)) {
        break;
      }

      // Collapsed spaces and stacks at the landing position block further
      // landings along this ray.
      if (boardManager.isCollapsedSpace(landingPos, board)) {
        break;
      }

      const landingStack = boardManager.getStack(landingPos, board);
      if (landingStack && landingStack.rings.length > 0) {
        break;
      }

      const candidate: Move = {
        id: '',
        type: 'overtaking_capture',
        player: playerNumber,
        from: position,
        captureTarget: targetPos,
        to: landingPos,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: gameState.moveHistory.length + 1
      };

      if (ruleEngine.validateMove(candidate, gameState)) {
        moves.push(candidate);
      }

      landingStep++;
    }
  }

  return moves;
}

/**
 * When multiple capture continuations are available from the current
 * chain position, use the generic PlayerChoice system to let the
 * active player choose a direction and landing.
 *
 * Mirrors GameEngine.chooseCaptureDirectionFromState but operates on
 * a passed-in ChainCaptureState.
 */
export async function chooseCaptureDirectionFromState(
  chainState: ChainCaptureState | undefined,
  gameState: GameState,
  deps: CaptureChainDeps
): Promise<Move | undefined> {
  if (!chainState) return undefined;

  const { boardManager, interactionManager } = deps;
  const options = chainState.availableMoves;
  if (options.length === 0) {
    return undefined;
  }

  // If there is no interaction manager or only one option, keep
  // behaviour simple and just return the sole available move.
  if (!interactionManager || options.length === 1) {
    return options[0];
  }

  const choice: CaptureDirectionChoice = {
    id: generateUUID(),
    gameId: gameState.id,
    playerNumber: chainState.playerNumber,
    type: 'capture_direction',
    prompt: 'Choose capture direction and landing position',
    options: options.map(opt => ({
      targetPosition: opt.captureTarget!,
      landingPosition: opt.to,
      capturedCapHeight:
        boardManager.getStack(opt.captureTarget!, gameState.board)?.capHeight || 0
    }))
  };

  const response: PlayerChoiceResponseFor<CaptureDirectionChoice> =
    await interactionManager.requestChoice(choice);
  const selected = response.selectedOption;

  const targetKey = positionToString(selected.targetPosition);
  const landingKey = positionToString(selected.landingPosition);

  // Find the matching Move in the available options; fall back to the
  // first option if for some reason we cannot match exactly.
  const matched = options.find(opt =>
    opt.captureTarget &&
    positionToString(opt.captureTarget) === targetKey &&
    positionToString(opt.to) === landingKey
  );

  return matched || options[0];
}

// Helper to get movement directions using BoardManager's config so we
// don't reach into GameEngine directly.
function getAllDirectionsForBoardType(
  _boardType: string,
  boardManager: BoardManager
): { x: number; y: number; z?: number }[] {
  const config = boardManager.getConfig();
  if (config.type === 'hexagonal') {
    return [
      { x: 1, y: 0, z: -1 },
      { x: 0, y: 1, z: -1 },
      { x: -1, y: 1, z: 0 },
      { x: -1, y: 0, z: 1 },
      { x: 0, y: -1, z: 1 },
      { x: 1, y: -1, z: 0 }
    ];
  }

  // square boards: Moore directions used for movement rays
  const directions: { x: number; y: number }[] = [];
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      if (dx === 0 && dy === 0) continue;
      directions.push({ x: dx, y: dy });
    }
  }
  return directions;
}

// Local UUID generator mirroring GameEngine.generateUUID
function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
