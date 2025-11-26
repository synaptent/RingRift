import type {
  GameState,
  Move,
  Position,
  CaptureDirectionChoice,
  PlayerChoiceResponseFor,
} from '../../../shared/engine';
import type { CaptureBoardAdapters } from '../../../shared/engine';
import { positionToString, enumerateCaptureMoves } from '../../../shared/engine';
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
    capturedCapHeight,
  };

  if (!state) {
    return {
      playerNumber: move.player,
      startPosition: move.from,
      currentPosition: move.to,
      segments: [segment],
      availableMoves: [],
      visitedPositions: new Set<string>([positionToString(move.from)]),
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
  const { boardManager } = deps;
  const board = gameState.board;
  const attackerStack = boardManager.getStack(position, board);

  if (!attackerStack || attackerStack.controllingPlayer !== playerNumber) {
    return [];
  }

  // Adapt the current board view to the shared capture enumerator so that
  // backend chain-capture enumeration stays in lock-step with the
  // sandbox and shared core rules.
  const adapters: CaptureBoardAdapters = {
    isValidPosition: (pos: Position) => boardManager.isValidPosition(pos),
    isCollapsedSpace: (pos: Position) => boardManager.isCollapsedSpace(pos, board),
    getStackAt: (pos: Position) => {
      const stack = boardManager.getStack(pos, board);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => boardManager.getMarker(pos, board),
  };

  const baseMoveNumber = gameState.moveHistory.length + 1;

  const rawMoves = enumerateCaptureMoves(
    gameState.boardType,
    position,
    playerNumber,
    adapters,
    baseMoveNumber
  );

  // Ensure stable-ish IDs for diagnostics and tests that may log moves;
  // geometry (from/target/to) is what parity tests actually care about.
  const moves: Move[] = rawMoves.map((m, index) => ({
    ...m,
    id:
      m.id && m.id.length > 0
        ? m.id
        : `capture-${positionToString(m.from!)}-${positionToString(
            m.captureTarget!
          )}-${positionToString(m.to!)}-${index}`,
    moveNumber: baseMoveNumber,
  }));

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
    id: generateUUID('capture_direction', gameState.id, chainState.segments.length, options.length),
    gameId: gameState.id,
    playerNumber: chainState.playerNumber,
    type: 'capture_direction',
    prompt: 'Choose capture direction and landing position',
    options: options.map((opt) => ({
      targetPosition: opt.captureTarget!,
      landingPosition: opt.to,
      capturedCapHeight: boardManager.getStack(opt.captureTarget!, gameState.board)?.capHeight || 0,
    })),
  };

  const response: PlayerChoiceResponseFor<CaptureDirectionChoice> =
    await interactionManager.requestChoice(choice);
  const selected = response.selectedOption;

  const targetKey = positionToString(selected.targetPosition);
  const landingKey = positionToString(selected.landingPosition);

  // Find the matching Move in the available options; fall back to the
  // first option if for some reason we cannot match exactly.
  const matched = options.find(
    (opt) =>
      opt.captureTarget &&
      positionToString(opt.captureTarget) === targetKey &&
      positionToString(opt.to) === landingKey
  );

  return matched || options[0];
}

// Local deterministic identifier helper for capture-direction choices.
// This deliberately avoids any RNG so that core rules behaviour remains
// fully deterministic (RR‑CANON R190). Callers should pass in structured
// context (e.g. game id, segment index, option count) so IDs remain
// unique and stable for parity/diagnostic tooling.
function generateUUID(...parts: Array<string | number | undefined>): string {
  return parts
    .filter((part) => part !== undefined)
    .map((part) => String(part))
    .join('|');
}
