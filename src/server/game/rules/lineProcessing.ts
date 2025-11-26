import type {
  GameState,
  Position,
  LineInfo,
  PlayerChoiceResponseFor,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
} from '../../../shared/engine';
import { BOARD_CONFIGS, positionToString, calculateCapHeight } from '../../../shared/engine';
import { BoardManager } from '../BoardManager';
import { PlayerInteractionManager } from '../PlayerInteractionManager';

export interface LineProcessingDeps {
  boardManager: BoardManager;
  interactionManager?: PlayerInteractionManager | undefined;
}

/**
 * Process all line formations with graduated rewards for the current player.
 *
 * This is a direct extraction of GameEngine.processLineFormations/
 * processOneLine/collapseLineMarkers/elimination helpers, rewritten in a
 * functional style but preserving semantics.
 */
export async function processLinesForCurrentPlayer(
  gameState: GameState,
  deps: LineProcessingDeps
): Promise<GameState> {
  const { boardManager, interactionManager } = deps;
  const config = BOARD_CONFIGS[gameState.boardType];

  // Keep processing until no more lines exist
  while (true) {
    const allLines = boardManager.findAllLines(gameState.board);
    if (allLines.length === 0) break;

    // Only consider lines for the moving player
    const playerLines = allLines.filter((line) => line.player === gameState.currentPlayer);
    if (playerLines.length === 0) break;

    let lineToProcess: LineInfo;

    if (!interactionManager || playerLines.length === 1) {
      // No interaction manager wired yet, or only one choice: keep current behaviour
      lineToProcess = playerLines[0];
    } else {
      const choice: LineOrderChoice = {
        id: generateUUID('line_order', gameState.id, gameState.history.length, playerLines.length),
        gameId: gameState.id,
        playerNumber: gameState.currentPlayer,
        type: 'line_order',
        prompt: 'Choose which line to process first',
        options: playerLines.map((line, index) => {
          const lineKey = line.positions.map((p) => positionToString(p)).join('|');
          return {
            lineId: String(index),
            markerPositions: line.positions,
            /**
             * Stable identifier for the canonical 'process_line' Move that
             * would process this line when enumerated via advanced-phase
             * helpers (RuleEngine.getValidMoves / GameEngine.getValidMoves
             * in the line_processing phase). This lets transports/AI map
             * this choice option directly onto a Move.id.
             */
            moveId: `process-line-${index}-${lineKey}`,
          };
        }),
      };

      const response: PlayerChoiceResponseFor<LineOrderChoice> =
        await interactionManager.requestChoice(choice);
      const selected = response.selectedOption;
      const index = parseInt(selected.lineId, 10);
      lineToProcess = playerLines[index] ?? playerLines[0];
    }

    gameState = await processOneLine(gameState, lineToProcess, config.lineLength, deps);
    // After processing one line, loop will re-evaluate remaining lines
  }

  return gameState;
}

/**
 * Process a single line formation for the given player.
 *
 * Rule Reference: Section 11.2
 */
async function processOneLine(
  gameState: GameState,
  line: LineInfo,
  requiredLength: number,
  deps: LineProcessingDeps
): Promise<GameState> {
  const lineLength = line.positions.length;

  if (lineLength === requiredLength) {
    // Exact required length: Must collapse all and eliminate ring/cap
    gameState = collapseLineMarkers(gameState, line.positions, line.player, deps);
    gameState = await eliminatePlayerRingOrCapWithChoice(gameState, line.player, deps);
    return gameState;
  }

  if (lineLength > requiredLength) {
    // Longer than required: player chooses Option 1 or Option 2 when an
    // interaction manager is available; otherwise, preserve current
    // behaviour and default to Option 2 (collapse minimum only, no elimination).
    const { interactionManager } = deps;

    if (!interactionManager) {
      const markersToCollapse = line.positions.slice(0, requiredLength);
      return collapseLineMarkers(gameState, markersToCollapse, line.player, deps);
    }

    const choice: LineRewardChoice = {
      id: generateUUID(
        'line_reward',
        gameState.id,
        gameState.history.length,
        line.positions.length
      ),
      gameId: gameState.id,
      playerNumber: gameState.currentPlayer,
      type: 'line_reward_option',
      prompt: 'Choose line reward option',
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    };

    const response: PlayerChoiceResponseFor<LineRewardChoice> =
      await interactionManager.requestChoice(choice);
    const selected = response.selectedOption;

    if (selected === 'option_1_collapse_all_and_eliminate') {
      gameState = collapseLineMarkers(gameState, line.positions, line.player, deps);
      gameState = await eliminatePlayerRingOrCapWithChoice(gameState, line.player, deps);
      return gameState;
    }

    const markersToCollapse = line.positions.slice(0, requiredLength);
    return collapseLineMarkers(gameState, markersToCollapse, line.player, deps);
  }

  return gameState;
}

/**
 * Collapse marker positions to player's color territory and update
 * territorySpaces. Mirrors GameEngine.collapseLineMarkers.
 */
function collapseLineMarkers(
  gameState: GameState,
  positions: Position[],
  playerNumber: number,
  deps: LineProcessingDeps
): GameState {
  const { boardManager } = deps;

  for (const pos of positions) {
    boardManager.setCollapsedSpace(pos, playerNumber, gameState.board);
  }

  // Update player's territory count
  gameState = updatePlayerTerritorySpaces(gameState, playerNumber, positions.length);
  return gameState;
}

/**
 * Eliminate one ring or cap from player's controlled stacks using the
 * player choice system when available. Falls back to default behaviour
 * when no interaction manager is wired. Mirrors
 * GameEngine.eliminatePlayerRingOrCapWithChoice.
 */
export async function eliminatePlayerRingOrCapWithChoice(
  gameState: GameState,
  player: number,
  deps: LineProcessingDeps
): Promise<GameState> {
  const { boardManager, interactionManager } = deps;
  const playerStacks = boardManager.getPlayerStacks(gameState.board, player);

  if (playerStacks.length === 0) {
    // Mirror the hand-elimination behaviour: eliminate from rings in hand
    const playerState = gameState.players.find((p) => p.playerNumber === player);
    if (playerState && playerState.ringsInHand > 0) {
      playerState.ringsInHand--;
      gameState.totalRingsEliminated++;
      if (!gameState.board.eliminatedRings[player]) {
        gameState.board.eliminatedRings[player] = 0;
      }
      gameState.board.eliminatedRings[player]++;
      gameState = updatePlayerEliminatedRings(gameState, player, 1);
    }
    return gameState;
  }

  if (!interactionManager || playerStacks.length === 1) {
    // No manager or only one stack: use default behaviour
    return eliminatePlayerRingOrCap(gameState, player, deps);
  }

  const choice: RingEliminationChoice = {
    id: generateUUID(
      'ring_elimination',
      gameState.id,
      gameState.history.length,
      playerStacks.length
    ),
    gameId: gameState.id,
    playerNumber: player,
    type: 'ring_elimination',
    prompt: 'Choose which stack to eliminate from',
    options: playerStacks.map((stack) => {
      const stackKey = positionToString(stack.position);
      return {
        stackPosition: stack.position,
        capHeight: stack.capHeight,
        totalHeight: stack.stackHeight,
        /**
         * Stable identifier for the canonical 'eliminate_rings_from_stack'
         * Move that would eliminate from this stack when enumerated via
         * advanced-phase helpers. This lets transports/AI map this choice
         * option directly onto a Move.id.
         */
        moveId: `eliminate-${stackKey}`,
      };
    }),
  };

  const response: PlayerChoiceResponseFor<RingEliminationChoice> =
    await interactionManager.requestChoice(choice);
  const selected = response.selectedOption;

  const selectedKey = positionToString(selected.stackPosition);
  const chosenStack =
    playerStacks.find((s) => positionToString(s.position) === selectedKey) || playerStacks[0];

  return eliminateFromStack(gameState, chosenStack, player, deps);
}

/**
 * Eliminate one ring or cap from player's controlled stacks.
 * Mirrors GameEngine.eliminatePlayerRingOrCap.
 */
function eliminatePlayerRingOrCap(
  gameState: GameState,
  player: number,
  deps: LineProcessingDeps
): GameState {
  const { boardManager } = deps;
  const playerStacks = boardManager.getPlayerStacks(gameState.board, player);

  if (playerStacks.length === 0) {
    const playerState = gameState.players.find((p) => p.playerNumber === player);
    if (playerState && playerState.ringsInHand > 0) {
      playerState.ringsInHand--;
      gameState.totalRingsEliminated++;

      if (!gameState.board.eliminatedRings[player]) {
        gameState.board.eliminatedRings[player] = 0;
      }
      gameState.board.eliminatedRings[player]++;

      gameState = updatePlayerEliminatedRings(gameState, player, 1);
    }
    return gameState;
  }

  const stack = playerStacks[0];
  return eliminateFromStack(gameState, stack, player, deps);
}

/**
 * Core elimination logic from a specific stack. Used by both the
 * default elimination path and the choice-based elimination helper.
 */
function eliminateFromStack(
  gameState: GameState,
  stack: ReturnType<BoardManager['getPlayerStacks']>[number],
  player: number,
  deps: LineProcessingDeps
): GameState {
  const { boardManager } = deps;

  // Calculate cap height
  const capHeight = calculateCapHeight(stack.rings);

  // Eliminate the entire cap (all consecutive top rings of controlling color)
  const remainingRings = stack.rings.slice(capHeight);

  // Update eliminated rings count
  gameState.totalRingsEliminated += capHeight;
  if (!gameState.board.eliminatedRings[player]) {
    gameState.board.eliminatedRings[player] = 0;
  }
  gameState.board.eliminatedRings[player] += capHeight;

  gameState = updatePlayerEliminatedRings(gameState, player, capHeight);

  if (remainingRings.length > 0) {
    // Update stack with remaining rings
    const newStack = {
      ...stack,
      rings: remainingRings,
      stackHeight: remainingRings.length,
      capHeight: calculateCapHeight(remainingRings),
      controllingPlayer: remainingRings[0],
    };
    boardManager.setStack(stack.position, newStack, gameState.board);
  } else {
    // Stack is now empty, remove it
    boardManager.removeStack(stack.position, gameState.board);
  }

  return gameState;
}

/**
 * Update player's eliminatedRings counter. Mirrors GameEngine.updatePlayerEliminatedRings.
 */
export function updatePlayerEliminatedRings(
  gameState: GameState,
  playerNumber: number,
  count: number
): GameState {
  const player = gameState.players.find((p) => p.playerNumber === playerNumber);
  if (player) {
    player.eliminatedRings += count;
  }
  return gameState;
}

/**
 * Update player's territorySpaces counter. Mirrors GameEngine.updatePlayerTerritorySpaces.
 */
export function updatePlayerTerritorySpaces(
  gameState: GameState,
  playerNumber: number,
  count: number
): GameState {
  const player = gameState.players.find((p) => p.playerNumber === playerNumber);
  if (player) {
    player.territorySpaces += count;
  }
  return gameState;
}

// Local deterministic identifier helper for line/territory-related choices.
// This deliberately avoids any RNG so that core rules behaviour remains
// fully deterministic (RR‑CANON R190). Callers pass structured context
// (game id, history length, candidate count, etc.) so IDs remain unique
// and stable for parity/diagnostic tooling.
function generateUUID(...parts: Array<string | number | undefined>): string {
  return parts
    .filter((part) => part !== undefined)
    .map((part) => String(part))
    .join('|');
}
