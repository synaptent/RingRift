import {
  GameState,
  Move,
  Player,
  BoardType,
  TimeControl,
  GameResult,
  BOARD_CONFIGS,
  Position,
  RingStack,
  Territory,
  LineInfo,
  positionToString,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  CaptureDirectionChoice,
  PlayerChoiceResponseFor,
  GameHistoryEntry,
} from '../../shared/types/game';
import {
  calculateCapHeight,
  getPathPositions,
  getMovementDirectionsForBoardType,
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
} from '../../shared/engine/core';
import { BoardManager } from './BoardManager';
import { RuleEngine } from './RuleEngine';
import { PlayerInteractionManager } from './PlayerInteractionManager';
import { processLinesForCurrentPlayer } from './rules/lineProcessing';
import { processDisconnectedRegionsForCurrentPlayer } from './rules/territoryProcessing';
import {
  ChainCaptureState,
  ChainCaptureSegment,
  updateChainCaptureStateAfterCapture as updateChainCaptureStateAfterCaptureShared,
  getCaptureOptionsFromPosition as getCaptureOptionsFromPositionShared,
  chooseCaptureDirectionFromState as chooseCaptureDirectionFromStateShared,
} from './rules/captureChainEngine';
import {
  PerTurnState,
  advanceGameForCurrentPlayer,
  updatePerTurnStateAfterMove as updatePerTurnStateAfterMoveTurn,
  TurnEngineDeps,
  TurnEngineHooks,
} from './turn/TurnEngine';

/**
 * Internal state for enforcing mandatory chain captures during the capture phase.
 *
 * This is intentionally kept out of the wire-level GameState so we can evolve
 * the representation without breaking clients. It is roughly modeled after the
 * Rust engine's `ChainCaptureState` and is used only inside GameEngine.
 *
 * The concrete shape is shared with the rules/captureChainEngine module; we
 * keep the Ts* aliases here to preserve existing semantics and comments while
 * centralising the implementation.
 */
type TsChainCaptureSegment = ChainCaptureSegment;
type TsChainCaptureState = ChainCaptureState;

// Timer functions for Node.js environment
declare const setTimeout: (callback: () => void, ms: number) => any;

declare const clearTimeout: (timer: any) => void;

// Using a simple UUID generator for now
function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export class GameEngine {
  private gameState: GameState;
  private boardManager: BoardManager;
  private ruleEngine: RuleEngine;
  private moveTimers: Map<number, any> = new Map();
  private interactionManager: PlayerInteractionManager | undefined;
  /**
   * Per-turn placement state: when a ring placement occurs, we track that
   * fact and remember which stack must be moved this turn. This mirrors
   * the sandbox engine’s per-turn fields but remains internal to the
   * backend engine.
   */
  private hasPlacedThisTurn: boolean = false;
  private mustMoveFromStackKey: string | undefined;
  /**
   * Internal chain capture state, used to enforce mandatory continuation of
   * captures once started. When defined, only additional overtaking captures
   * from `currentPosition` are legal until no options remain.
   */
  private chainCaptureState: TsChainCaptureState | undefined;

  constructor(
    gameId: string,
    boardType: BoardType,
    players: Player[],
    timeControl: TimeControl,
    isRated: boolean = true,
    interactionManager?: PlayerInteractionManager
  ) {
    this.boardManager = new BoardManager(boardType);
    this.ruleEngine = new RuleEngine(this.boardManager, boardType);
    this.interactionManager = interactionManager;

    const config = BOARD_CONFIGS[boardType];

    this.gameState = {
      id: gameId,
      boardType,
      board: this.boardManager.createBoard(),
      players: players.map((p, index) => ({
        ...p,
        playerNumber: index + 1,
        timeRemaining: timeControl.initialTime * 1000, // Convert to milliseconds
        isReady: p.type === 'ai', // AI players are always ready
      })),
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl,
      spectators: [],
      gameStatus: 'waiting',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated,
      maxPlayers: players.length,
      totalRingsInPlay: config.ringsPerPlayer * players.length,
      totalRingsEliminated: 0,
      victoryThreshold: Math.floor((config.ringsPerPlayer * players.length) / 2) + 1,
      territoryVictoryThreshold: Math.floor(config.totalSpaces / 2) + 1,
    };
  }

  getGameState(): GameState {
    const state = this.gameState;

    // Deep-clone the board and key collections so tests (especially the
    // AI simulation debug harness) see true pre/post snapshots rather
    // than views that can be mutated via shared Maps.
    const board = state.board;
    const clonedBoard = {
      ...board,
      stacks: new Map(board.stacks),
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      territories: new Map(board.territories),
      formedLines: [...board.formedLines],
      eliminatedRings: { ...board.eliminatedRings },
    };

    return {
      ...state,
      board: clonedBoard,
      moveHistory: [...state.moveHistory],
      history: [...state.history],
      players: state.players.map((p) => ({ ...p })),
      spectators: [...state.spectators],
    };
  }

  /**
   * Helper to ensure that a PlayerInteractionManager is available when
   * attempting to perform any player-facing choice. This keeps the core
   * engine decoupled from transport/UI while still enforcing that choices
   * cannot be resolved silently once integration is enabled.
   */
  private requireInteractionManager(): PlayerInteractionManager {
    if (!this.interactionManager) {
      throw new Error('PlayerInteractionManager is required for player choice operations');
    }
    return this.interactionManager;
  }

  startGame(): boolean {
    // Check if all players are ready
    const allReady = this.gameState.players.every((p) => p.isReady);
    if (!allReady) {
      return false;
    }

    this.gameState.gameStatus = 'active';
    this.gameState.lastMoveAt = new Date();

    // Start the first player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);

    return true;
  }

  /**
   * Append a structured history entry for a canonical move applied to the
   * engine. This is the primary hook used by parity/debug tooling; it is
   * intentionally side-effect-free with respect to core rules logic.
   */
  private appendHistoryEntry(before: GameState, action: Move): void {
    const after = this.getGameState();

    const progressBefore = computeProgressSnapshot(before);
    const progressAfter = computeProgressSnapshot(after);

    const entry: GameHistoryEntry = {
      moveNumber: action.moveNumber,
      action,
      actor: action.player,
      phaseBefore: before.currentPhase,
      phaseAfter: after.currentPhase,
      statusBefore: before.gameStatus,
      statusAfter: after.gameStatus,
      progressBefore,
      progressAfter,
      stateHashBefore: hashGameState(before),
      stateHashAfter: hashGameState(after),
      boardBeforeSummary: summarizeBoard(before.board),
      boardAfterSummary: summarizeBoard(after.board),
    };

    const history: GameHistoryEntry[] = [...this.gameState.history, entry];
    this.gameState = {
      ...this.gameState,
      history,
    };
  }

  async makeMove(move: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>): Promise<{
    success: boolean;
    error?: string;
    gameState?: GameState;
    gameResult?: GameResult;
  }> {
    // When a chain capture is in progress, only additional overtaking captures
    // from the current chain position are legal until no options remain.
    if (this.chainCaptureState) {
      const state = this.chainCaptureState;

      // All moves during a chain must be made by the same player.
      if (move.player !== state.playerNumber) {
        return {
          success: false,
          error: 'Chain capture in progress: only the capturing player may move',
        };
      }

      // During a chain, only overtaking_capture moves from the current position
      // are allowed. Ending a chain early is not permitted by the rules; the
      // chain ends only when no further captures are available.
      if (
        move.type !== 'overtaking_capture' ||
        !move.from ||
        positionToString(move.from) !== positionToString(state.currentPosition)
      ) {
        return {
          success: false,
          error: 'Chain capture in progress: must continue capturing with the same stack',
        };
      }
    }

    // Enforce must-move origin when a placement has occurred this turn.
    if (this.mustMoveFromStackKey) {
      const moveFromKey = move.from ? positionToString(move.from) : undefined;

      const isMovementOrCaptureType =
        move.type === 'move_stack' ||
        move.type === 'move_ring' ||
        move.type === 'build_stack' ||
        move.type === 'overtaking_capture';

      if (isMovementOrCaptureType && (!moveFromKey || moveFromKey !== this.mustMoveFromStackKey)) {
        return {
          success: false,
          error: 'You must move the stack that was just placed or updated this turn',
        };
      }
    }

    // Capture a pre-move snapshot for history/event-sourcing. This uses the
    // public getGameState() so callers and history entries share the same
    // cloned view of board/maps.
    const beforeStateForHistory = this.getGameState();

    // Validate the move at the rules level
    const fullMove: Move = {
      ...move,
      id: generateUUID(),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: this.gameState.moveHistory.length + 1,
    };

    const validation = this.ruleEngine.validateMove(fullMove, this.gameState);
    if (!validation) {
      return {
        success: false,
        error: 'Invalid move',
      };
    }

    // Defensive runtime check: for movement and capture moves, verify that
    // the source stack actually exists and is controlled by this player.
    // This catches stale moves where validation passed on an earlier state
    // but the stack is now missing (e.g., due to intervening auto phases or
    // concurrent state changes).
    const isMovementOrCaptureType =
      fullMove.type === 'move_stack' ||
      fullMove.type === 'move_ring' ||
      fullMove.type === 'overtaking_capture';

    if (isMovementOrCaptureType && fullMove.from) {
      const sourceStack = this.boardManager.getStack(fullMove.from, this.gameState.board);
      if (!sourceStack) {
        console.error(
          '[GameEngine.makeMove] S-invariant violation: move validated but source stack missing',
          {
            moveType: fullMove.type,
            player: fullMove.player,
            from: positionToString(fullMove.from),
            to: fullMove.to ? positionToString(fullMove.to) : undefined,
            currentPhase: this.gameState.currentPhase,
            currentPlayer: this.gameState.currentPlayer,
          }
        );
        return {
          success: false,
          error: 'Source stack no longer exists',
        };
      }
      if (sourceStack.controllingPlayer !== fullMove.player) {
        console.error(
          '[GameEngine.makeMove] S-invariant violation: move validated but source stack not controlled by player',
          {
            moveType: fullMove.type,
            player: fullMove.player,
            sourceControllingPlayer: sourceStack.controllingPlayer,
            from: positionToString(fullMove.from),
            to: fullMove.to ? positionToString(fullMove.to) : undefined,
            currentPhase: this.gameState.currentPhase,
            currentPlayer: this.gameState.currentPlayer,
          }
        );
        return {
          success: false,
          error: 'Source stack is not controlled by this player',
        };
      }
    }

    // Capture context needed for chain state bookkeeping (cap height, etc.)
    let capturedCapHeight = 0;
    if (fullMove.type === 'overtaking_capture' && fullMove.captureTarget) {
      const targetStack = this.boardManager.getStack(fullMove.captureTarget, this.gameState.board);
      capturedCapHeight = targetStack ? targetStack.capHeight : 0;
    }

    // Stop current player's timer while we process the move
    this.stopPlayerTimer(this.gameState.currentPlayer);

    // Apply the move to the board state. We intentionally ignore the
    // granular result here; post-move consequences (lines, territory,
    // etc.) are processed separately based on the updated gameState.
    this.applyMove(fullMove);

    // Add move to history
    this.gameState.moveHistory.push(fullMove);
    this.gameState.lastMoveAt = new Date();

    // Update per-turn placement/movement bookkeeping so that subsequent
    // phases (movement/capture) can enforce must-move constraints.
    this.updatePerTurnStateAfterMove(fullMove);

    // If this was an overtaking capture, update or start the chain
    // capture state and, if additional captures are available, drive
    // the rest of the chain from within the engine. This includes
    // invoking CaptureDirectionChoice via PlayerInteractionManager
    // when multiple follow-up options exist.
    if (fullMove.type === 'overtaking_capture') {
      this.updateChainCaptureStateAfterCapture(fullMove, capturedCapHeight);

      // Engine-driven chain continuation loop
      while (true) {
        const state = this.chainCaptureState;
        const currentPlayer = this.gameState.currentPlayer;

        if (!state || state.playerNumber !== currentPlayer) {
          // No active chain (or somehow not this player's chain): stop.
          this.chainCaptureState = undefined;
          break;
        }

        const followUpMoves = this.getCaptureOptionsFromPosition(
          state.currentPosition,
          currentPlayer
        );
        state.availableMoves = followUpMoves;

        if (followUpMoves.length === 0) {
          // Chain is exhausted; clear state and exit loop.
          this.chainCaptureState = undefined;
          break;
        }

        // Let the player choose among available capture directions when
        // appropriate; this uses the CaptureDirectionChoice flow.
        const nextChosen = await this.chooseCaptureDirectionFromState();
        if (!nextChosen) {
          // Defensive: if no choice can be made, end the chain.
          this.chainCaptureState = undefined;
          break;
        }

        // Compute cap height for the next target, primarily for
        // diagnostic/state-tracking parity with the Rust engine.
        let nextCapturedCapHeight = 0;
        if (nextChosen.captureTarget) {
          const nextTargetStack = this.boardManager.getStack(
            nextChosen.captureTarget,
            this.gameState.board
          );
          nextCapturedCapHeight = nextTargetStack ? nextTargetStack.capHeight : 0;
        }

        // Promote the chosen follow-up capture into a full Move with
        // its own id/timestamp and append it to history. These
        // internal chain segments remain part of the same turn.
        const internalMove: Move = {
          ...nextChosen,
          id: generateUUID(),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: this.gameState.moveHistory.length + 1,
        };

        this.applyMove(internalMove);
        this.gameState.moveHistory.push(internalMove);
        this.gameState.lastMoveAt = new Date();

        // Update chain state for the new position and continue loop.
        this.updateChainCaptureStateAfterCapture(internalMove, nextCapturedCapHeight);
      }
    } else {
      // Any non-capture move clears any stale chain state (defensive safety).
      this.chainCaptureState = undefined;
    }

    // Process automatic consequences (line formations, territory, etc.) only
    // after the full move (including any mandatory chain) has resolved.
    await this.processAutomaticConsequences();

    // Check for game end conditions
    const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
    if (gameEndCheck.isGameOver) {
      return this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'unknown');
    }

    // Advance to next phase/player
    this.advanceGame();

    // Step through automatic bookkeeping phases (line_processing and
    // territory_processing) so the post-move snapshot and history entry
    // reflect the same next-player interactive phase that the sandbox
    // engine records in its traces.
    this.stepAutomaticPhasesForTesting();

    // Start next player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);

    // Record a structured history entry for this canonical move.
    this.appendHistoryEntry(beforeStateForHistory, fullMove);

    return {
      success: true,
      gameState: this.getGameState(),
    };
  }

  private applyMove(move: Move): {
    captures: Position[];
    territoryChanges: Territory[];
    lineCollapses: LineInfo[];
  } {
    const result = {
      captures: [] as Position[],
      territoryChanges: [] as Territory[],
      lineCollapses: [] as LineInfo[],
    };

    switch (move.type) {
      case 'place_ring':
        if (move.to) {
          const board = this.gameState.board;
          const existingStack = this.boardManager.getStack(move.to, board);
          const placementCount = Math.max(1, move.placementCount ?? 1);

          const placementRings = new Array(placementCount).fill(move.player);

          let newRings: number[];
          if (existingStack && existingStack.rings.length > 0) {
            // Placing on an existing stack: new rings sit on top
            newRings = [...placementRings, ...existingStack.rings];
          } else {
            // Placing on an empty space
            newRings = placementRings;
          }

          const newStack: RingStack = {
            position: move.to,
            rings: newRings,
            stackHeight: newRings.length,
            capHeight: calculateCapHeight(newRings),
            controllingPlayer: newRings[0],
          };

          this.boardManager.setStack(move.to, newStack, board);

          // Update player state: decrement rings in hand by placementCount,
          // clamped defensively to avoid going below zero.
          const player = this.gameState.players.find((p) => p.playerNumber === move.player);
          if (player && player.ringsInHand > 0) {
            const toSpend = Math.min(placementCount, player.ringsInHand);
            player.ringsInHand -= toSpend;
          }
        }
        break;

      case 'skip_placement':
        // No-op at the board level. The RuleEngine has already verified
        // that skipping is only allowed when placement is optional, so
        // we simply advance to movement via advanceGame() without
        // modifying board or per-player ring counts.
        break;

      case 'move_ring':
      case 'move_stack':
        if (move.from && move.to) {
          const stack = this.boardManager.getStack(move.from, this.gameState.board);
          if (!stack) {
            // DIAGNOSTIC: This should never happen if validation and defensive
            // checks are working correctly. Log and bail to prevent silent no-ops.
            console.error('[GameEngine.applyMove] BUG: move_stack/move_ring but no source stack', {
              moveType: move.type,
              player: move.player,
              from: positionToString(move.from),
              to: positionToString(move.to),
              availableStacks: Array.from(this.gameState.board.stacks.keys()),
            });
            break; // Early exit from switch - no state change
          }

          // Rule Reference: Section 4.2.1 - Leave marker on departure space
          this.boardManager.setMarker(move.from, move.player, this.gameState.board);

          // Process markers along movement path (Section 8.3)
          this.processMarkersAlongPath(move.from, move.to, move.player);

          // Check if landing on same-color marker (Section 8.2 / 8.3.1)
          const landingMarker = this.boardManager.getMarker(move.to, this.gameState.board);
          const landedOnOwnMarker = landingMarker === move.player;
          if (landedOnOwnMarker) {
            // Stacks cannot coexist with markers; remove the marker prior
            // to landing, then apply the self-elimination rule below.
            this.boardManager.removeMarker(move.to, this.gameState.board);
          }

          // Remove stack from source
          this.boardManager.removeStack(move.from, this.gameState.board);

          // Normal movement (no capture at landing position)
          const movedStack: RingStack = {
            ...stack,
            position: move.to,
          };
          this.boardManager.setStack(move.to, movedStack, this.gameState.board);

          if (landedOnOwnMarker) {
            // New rule: landing on your own marker with a non-capture move
            // removes that marker and immediately eliminates your top ring,
            // credited toward ring-elimination victory conditions.
            this.eliminateTopRingAt(move.to, move.player);
          }
        }
        break;

      case 'overtaking_capture':
        if (move.from && move.to && move.captureTarget) {
          this.performOvertakingCapture(move.from, move.captureTarget, move.to, move.player);
          result.captures.push(move.captureTarget);
        }
        break;

      case 'build_stack':
        if (move.from && move.to && move.buildAmount) {
          const sourceStack = this.boardManager.getStack(move.from, this.gameState.board);
          const targetStack = this.boardManager.getStack(move.to, this.gameState.board);

          if (sourceStack && targetStack && move.buildAmount) {
            // Transfer rings from source to target
            const transferRings = sourceStack.rings.slice(0, move.buildAmount);
            const remainingRings = sourceStack.rings.slice(move.buildAmount);

            const newSourceStack: RingStack = {
              ...sourceStack,
              stackHeight: sourceStack.stackHeight - move.buildAmount,
              rings: remainingRings,
            };

            const newTargetStack: RingStack = {
              ...targetStack,
              stackHeight: targetStack.stackHeight + move.buildAmount,
              capHeight: Math.max(targetStack.capHeight, move.buildAmount),
              rings: [...targetStack.rings, ...transferRings],
            };

            // Update stacks
            if (newSourceStack.stackHeight > 0) {
              this.boardManager.setStack(move.from, newSourceStack, this.gameState.board);
            } else {
              this.boardManager.removeStack(move.from, this.gameState.board);
            }
            this.boardManager.setStack(move.to, newTargetStack, this.gameState.board);
          }
        }
        break;
    }

    // Line formation is processed separately in line_processing phase
    // This is just for tracking lines that were formed during the move

    // Check for territory disconnection
    const territories = this.boardManager.findAllTerritoriesForAllPlayers(this.gameState.board);
    for (const territory of territories) {
      if (territory.isDisconnected) {
        // Remove disconnected territory
        for (const pos of territory.spaces) {
          this.boardManager.removeStack(pos, this.gameState.board);
        }
        result.territoryChanges.push(territory);
      }
    }

    return result;
  }

  /**
   * Update or initialize the internal chain capture state after an
   * overtaking capture has been successfully applied to the board.
   *
   * This mirrors the Rust engine's ChainCaptureState at a high level:
   * we track the start position, current position, and the sequence of
   * capture segments taken so far.
   */
  private updateChainCaptureStateAfterCapture(move: Move, capturedCapHeight: number): void {
    this.chainCaptureState = updateChainCaptureStateAfterCaptureShared(
      this.chainCaptureState,
      move,
      capturedCapHeight
    );
  }

  /**
   * Enumerate all valid capture moves from a given position for the
   * specified player by ray-walking in each movement direction and
   * validating each candidate via RuleEngine.
   *
   * This is intentionally kept in sync with the Rust
   * CaptureProcessor::get_available_capture_details logic and is
   * exercised by tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts
   * as the TS reference for multi-option chain capture behavior.
   */
  private getCaptureOptionsFromPosition(position: Position, playerNumber: number): Move[] {
    return getCaptureOptionsFromPositionShared(position, playerNumber, this.gameState, {
      boardManager: this.boardManager,
      ruleEngine: this.ruleEngine,
      interactionManager: this.interactionManager,
    });
  }

  /**
   * When multiple capture continuations are available from the current
   * chain position, use the generic PlayerChoice system to let the
   * active player choose a direction and landing. This wires the
   * CaptureDirectionChoice type into GameEngine using the
   * chainCaptureState.availableMoves list.
   *
   * NOTE: This helper does not yet automatically apply the chosen move;
   * it simply selects and returns it. Future work can integrate this
   * into a full chain-capture loop once the transport/UI flow is ready.
   */
  private async chooseCaptureDirectionFromState(): Promise<Move | undefined> {
    return chooseCaptureDirectionFromStateShared(this.chainCaptureState, this.gameState, {
      boardManager: this.boardManager,
      ruleEngine: this.ruleEngine,
      interactionManager: this.interactionManager,
    });
  }

  /**
   * Core overtaking capture operation used for both user-initiated
   * captures (via applyMove) and engine-driven chain captures.
   * Rule Reference: Section 10.2 - Overtaking Capture
   */

  private performOvertakingCapture(
    from: Position,
    captureTarget: Position,
    landing: Position,
    player: number
  ): void {
    const stack = this.boardManager.getStack(from, this.gameState.board);
    const targetStack = this.boardManager.getStack(captureTarget, this.gameState.board);

    if (!stack || !targetStack) {
      return;
    }

    // Leave marker on departure space
    this.boardManager.setMarker(from, player, this.gameState.board);

    // Process markers along path to target
    this.processMarkersAlongPath(from, captureTarget, player);

    // Process markers along path from target to landing
    this.processMarkersAlongPath(captureTarget, landing, player);

    // Check if landing on a marker before resolving the capture. Any marker
    // present at the landing cell must be removed prior to placing the
    // capturing stack so that stacks and markers never coexist on the same
    // space. If the marker belongs to the capturing player, we also apply
    // the self-elimination rule after landing.
    const landingMarkerPlayer = this.boardManager.getMarker(landing, this.gameState.board);
    const landedOnOwnMarker = landingMarkerPlayer === player;
    if (landingMarkerPlayer !== undefined) {
      // Remove the marker prior to landing; the self-elimination rule, when
      // applicable, will be applied after the capturing stack is placed.
      this.boardManager.removeMarker(landing, this.gameState.board);
    }

    // Capture top ring from target stack and add to bottom of capturing stack
    const capturedRing = targetStack.rings[0]; // Top ring
    const newRings = [...stack.rings, capturedRing];

    // Update target stack (remove top ring)
    const remainingTargetRings = targetStack.rings.slice(1);
    if (remainingTargetRings.length > 0) {
      const newTargetStack: RingStack = {
        ...targetStack,
        rings: remainingTargetRings,
        stackHeight: remainingTargetRings.length,
        capHeight: calculateCapHeight(remainingTargetRings),
        controllingPlayer: remainingTargetRings[0],
      };
      this.boardManager.setStack(captureTarget, newTargetStack, this.gameState.board);
    } else {
      // Target stack is now empty, remove it
      this.boardManager.removeStack(captureTarget, this.gameState.board);
    }

    // Remove capturing stack from source
    this.boardManager.removeStack(from, this.gameState.board);

    // Place capturing stack at landing position with captured ring
    const newStack: RingStack = {
      position: landing,
      rings: newRings,
      stackHeight: newRings.length,
      capHeight: calculateCapHeight(newRings),
      controllingPlayer: newRings[0],
    };
    this.boardManager.setStack(landing, newStack, this.gameState.board);

    if (landedOnOwnMarker) {
      // New rule: landing on your own marker during an overtaking capture
      // removes that marker and immediately eliminates your top ring,
      // credited toward ring-elimination victory conditions.
      this.eliminateTopRingAt(landing, player);
    }
  }

  /**
   * Process automatic consequences after a move
   * Rule Reference: Section 4.5 - Post-Movement Processing
   */
  private async processAutomaticConsequences(): Promise<void> {
    // Captures are already processed in applyMove

    // Process line formations (Section 11.2, 11.3)
    this.gameState = await processLinesForCurrentPlayer(this.gameState, {
      boardManager: this.boardManager,
      interactionManager: this.interactionManager,
    });

    // Process territory disconnections (Section 12.2)
    this.gameState = await processDisconnectedRegionsForCurrentPlayer(this.gameState, {
      boardManager: this.boardManager,
      interactionManager: this.interactionManager,
    });
  }

  /**
   * Process all line formations with graduated rewards
   * Rule Reference: Section 11.2, 11.3
   *
   * For exact required length (4 for 8x8, 5 for 19x19/hex):
   *   - Collapse all markers
   *   - Eliminate one ring or cap from controlled stack
   *
   * For longer lines (5+ for 8x8, 6+ for 19x19/hex):
   *   - Option 1: Collapse all + eliminate ring/cap
   *   - Option 2: Collapse required markers only, no elimination
   */
  private async processLineFormations(): Promise<void> {
    const config = BOARD_CONFIGS[this.gameState.boardType];

    // Keep processing until no more lines exist
    while (true) {
      const allLines = this.boardManager.findAllLines(this.gameState.board);
      if (allLines.length === 0) break;

      // Only consider lines for the moving player
      const playerLines = allLines.filter((line) => line.player === this.gameState.currentPlayer);
      if (playerLines.length === 0) break;

      let lineToProcess: LineInfo;

      if (!this.interactionManager || playerLines.length === 1) {
        // No interaction manager wired yet, or only one choice: keep current behaviour
        lineToProcess = playerLines[0];
      } else {
        const interaction = this.requireInteractionManager();

        const choice: LineOrderChoice = {
          id: generateUUID(),
          gameId: this.gameState.id,
          playerNumber: this.gameState.currentPlayer,
          type: 'line_order',
          prompt: 'Choose which line to process first',
          options: playerLines.map((line, index) => ({
            lineId: String(index),
            markerPositions: line.positions,
          })),
        };

        const response: PlayerChoiceResponseFor<LineOrderChoice> =
          await interaction.requestChoice(choice);
        const selected = response.selectedOption;
        const index = parseInt(selected.lineId, 10);
        lineToProcess = playerLines[index] ?? playerLines[0];
      }

      await this.processOneLine(lineToProcess, config.lineLength);
      // After processing one line, loop will re-evaluate remaining lines
    }
  }

  /**
   * Process a single line formation
   * Rule Reference: Section 11.2
   */
  private async processOneLine(line: LineInfo, requiredLength: number): Promise<void> {
    const lineLength = line.positions.length;

    if (lineLength === requiredLength) {
      // Exact required length: Must collapse all and eliminate ring/cap
      this.collapseLineMarkers(line.positions, line.player);
      await this.eliminatePlayerRingOrCapWithChoice(line.player);
    } else if (lineLength > requiredLength) {
      // Longer than required: player chooses Option 1 or Option 2 when an
      // interaction manager is available; otherwise, preserve current
      // behaviour and default to Option 2 (collapse minimum only, no elimination).
      if (!this.interactionManager) {
        const markersToCollapse = line.positions.slice(0, requiredLength);
        this.collapseLineMarkers(markersToCollapse, line.player);
        return;
      }

      const interaction = this.requireInteractionManager();

      const choice: LineRewardChoice = {
        id: generateUUID(),
        gameId: this.gameState.id,
        playerNumber: this.gameState.currentPlayer,
        type: 'line_reward_option',
        prompt: 'Choose line reward option',
        options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
      };

      const response: PlayerChoiceResponseFor<LineRewardChoice> =
        await interaction.requestChoice(choice);
      const selected = response.selectedOption;

      if (selected === 'option_1_collapse_all_and_eliminate') {
        this.collapseLineMarkers(line.positions, line.player);
        await this.eliminatePlayerRingOrCapWithChoice(line.player);
      } else {
        const markersToCollapse = line.positions.slice(0, requiredLength);
        this.collapseLineMarkers(markersToCollapse, line.player);
      }
    }
  }

  /**
   * Collapse marker positions to player's color territory
   * Rule Reference: Section 11.2 - Markers collapse to colored spaces
   */
  private collapseLineMarkers(positions: Position[], player: number): void {
    for (const pos of positions) {
      this.boardManager.setCollapsedSpace(pos, player, this.gameState.board);
    }
    // Update player's territory count
    this.updatePlayerTerritorySpaces(player, positions.length);
  }

  /**
   * Eliminate one ring or cap from player's controlled stacks
   * Rule Reference: Section 11.2 - Moving player chooses which ring/stack cap to eliminate
   */
  private eliminatePlayerRingOrCap(player: number): void {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player);

    if (playerStacks.length === 0) {
      // No stacks to eliminate from, player might have rings in hand
      const playerState = this.gameState.players.find((p) => p.playerNumber === player);
      if (playerState && playerState.ringsInHand > 0) {
        // Eliminate from hand
        playerState.ringsInHand--;
        this.gameState.totalRingsEliminated++;

        // Track eliminated rings in board state
        if (!this.gameState.board.eliminatedRings[player]) {
          this.gameState.board.eliminatedRings[player] = 0;
        }
        this.gameState.board.eliminatedRings[player]++;

        // Update player state
        this.updatePlayerEliminatedRings(player, 1);
      }
      return;
    }

    // Default behaviour: eliminate from first stack
    const stack = playerStacks[0];
    this.eliminateFromStack(stack, player);
  }

  /**
   * Core elimination logic from a specific stack. Used by both the
   * default elimination path and the choice-based elimination helper.
   */
  private eliminateFromStack(stack: RingStack, player: number): void {
    // Calculate cap height
    const capHeight = calculateCapHeight(stack.rings);

    // Eliminate the entire cap (all consecutive top rings of controlling color)
    const remainingRings = stack.rings.slice(capHeight);

    // Update eliminated rings count
    this.gameState.totalRingsEliminated += capHeight;
    if (!this.gameState.board.eliminatedRings[player]) {
      this.gameState.board.eliminatedRings[player] = 0;
    }
    this.gameState.board.eliminatedRings[player] += capHeight;

    // Update player state
    this.updatePlayerEliminatedRings(player, capHeight);

    if (remainingRings.length > 0) {
      // Update stack with remaining rings
      const newStack: RingStack = {
        ...stack,
        rings: remainingRings,
        stackHeight: remainingRings.length,
        capHeight: calculateCapHeight(remainingRings),
        controllingPlayer: remainingRings[0],
      };
      this.boardManager.setStack(stack.position, newStack, this.gameState.board);
    } else {
      // Stack is now empty, remove it
      this.boardManager.removeStack(stack.position, this.gameState.board);
    }
  }

  /**
   * Eliminate exactly the top ring from the stack at the given position,
   * crediting the elimination to the specified player. This is used for
   * the "landing on your own marker eliminates your top ring" rule,
   * which applies to both non-capture moves and overtaking capture
   * segments.
   */
  private eliminateTopRingAt(position: Position, creditedPlayer: number): void {
    const stack = this.boardManager.getStack(position, this.gameState.board);
    if (!stack || stack.stackHeight === 0) {
      return;
    }

    // Remove the single top ring from the stack.
    const [, ...remainingRings] = stack.rings;

    // Update global elimination counters (one ring credited to the mover).
    this.gameState.totalRingsEliminated += 1;
    if (!this.gameState.board.eliminatedRings[creditedPlayer]) {
      this.gameState.board.eliminatedRings[creditedPlayer] = 0;
    }
    this.gameState.board.eliminatedRings[creditedPlayer] += 1;

    // Update per-player elimination stats.
    this.updatePlayerEliminatedRings(creditedPlayer, 1);

    if (remainingRings.length > 0) {
      const newStack: RingStack = {
        ...stack,
        rings: remainingRings,
        stackHeight: remainingRings.length,
        capHeight: calculateCapHeight(remainingRings),
        controllingPlayer: remainingRings[0],
      };
      this.boardManager.setStack(position, newStack, this.gameState.board);
    } else {
      // If no rings remain, remove the stack entirely.
      this.boardManager.removeStack(position, this.gameState.board);
    }
  }

  /**
   * Eliminate one ring or cap using the player choice system when available.
   * Falls back to default behaviour when no interaction manager is wired.
   */
  private async eliminatePlayerRingOrCapWithChoice(player: number): Promise<void> {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player);

    if (playerStacks.length === 0) {
      // Mirror the hand-elimination behaviour from eliminatePlayerRingOrCap
      const playerState = this.gameState.players.find((p) => p.playerNumber === player);
      if (playerState && playerState.ringsInHand > 0) {
        playerState.ringsInHand--;
        this.gameState.totalRingsEliminated++;
        if (!this.gameState.board.eliminatedRings[player]) {
          this.gameState.board.eliminatedRings[player] = 0;
        }
        this.gameState.board.eliminatedRings[player]++;
        this.updatePlayerEliminatedRings(player, 1);
      }
      return;
    }

    if (!this.interactionManager || playerStacks.length === 1) {
      // No manager or only one stack: use default behaviour
      this.eliminatePlayerRingOrCap(player);
      return;
    }

    const interaction = this.requireInteractionManager();

    const choice: RingEliminationChoice = {
      id: generateUUID(),
      gameId: this.gameState.id,
      playerNumber: player,
      type: 'ring_elimination',
      prompt: 'Choose which stack to eliminate from',
      options: playerStacks.map((stack) => ({
        stackPosition: stack.position,
        capHeight: stack.capHeight,
        totalHeight: stack.stackHeight,
      })),
    };

    const response: PlayerChoiceResponseFor<RingEliminationChoice> =
      await interaction.requestChoice(choice);
    const selected = response.selectedOption;

    const selectedKey = positionToString(selected.stackPosition);
    const chosenStack =
      playerStacks.find((s) => positionToString(s.position) === selectedKey) || playerStacks[0];

    this.eliminateFromStack(chosenStack, player);
  }

  /**
   * Update player's eliminatedRings counter
   */
  private updatePlayerEliminatedRings(playerNumber: number, count: number): void {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (player) {
      player.eliminatedRings += count;
    }
  }

  /**
   * Update player's territorySpaces counter
   */
  private updatePlayerTerritorySpaces(playerNumber: number, count: number): void {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (player) {
      player.territorySpaces += count;
    }
  }

  /**
   * Process disconnected regions with chain reactions
   * Rule Reference: Section 12.2, 12.3 - Territory Disconnection and Chain Reactions
   */
  private async processDisconnectedRegions(): Promise<void> {
    const movingPlayer = this.gameState.currentPlayer;

    // Keep processing until no more disconnections occur
    while (true) {
      const disconnectedRegions = this.boardManager.findDisconnectedRegions(
        this.gameState.board,
        movingPlayer
      );

      if (disconnectedRegions.length === 0) break;

      // Filter to regions that satisfy the self-elimination prerequisite
      // for the moving player. This mirrors the Rust notion of
      // "eligible" disconnected regions and prevents us from
      // prematurely bailing out just because the first region is not
      // processable.
      const eligibleRegions = disconnectedRegions.filter((region) =>
        this.canProcessDisconnectedRegion(region, movingPlayer)
      );

      if (eligibleRegions.length === 0) {
        // No region can be processed for this player; stop to avoid
        // infinite loops.
        break;
      }

      let region: Territory;

      if (!this.interactionManager || eligibleRegions.length === 1) {
        // No manager or only one eligible region: process it directly.
        region = eligibleRegions[0];
      } else {
        const interaction = this.requireInteractionManager();
        const choice: RegionOrderChoice = {
          id: generateUUID(),
          gameId: this.gameState.id,
          playerNumber: movingPlayer,
          type: 'region_order',
          prompt: 'Choose which disconnected region to process first',
          options: eligibleRegions.map((r, index) => ({
            regionId: String(index),
            size: r.spaces.length,
            representativePosition: r.spaces[0],
          })),
        };

        const response: PlayerChoiceResponseFor<RegionOrderChoice> =
          await interaction.requestChoice(choice);
        const selected = response.selectedOption;
        const index = parseInt(selected.regionId, 10);
        region = eligibleRegions[index] ?? eligibleRegions[0];
      }

      // Process the disconnected region
      await this.processOneDisconnectedRegion(region, movingPlayer);
    }
  }

  /**
   * Check if player can process a disconnected region
   * Rule Reference: Section 12.2 - Self-Elimination Prerequisite
   *
   * Player must have at least one ring/cap outside the region before processing
   */
  private canProcessDisconnectedRegion(region: Territory, player: number): boolean {
    const regionPositionSet = new Set(region.spaces.map((pos) => positionToString(pos)));
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player);

    // Check if player has at least one ring/cap outside this region
    for (const stack of playerStacks) {
      const stackPosKey = positionToString(stack.position);
      if (!regionPositionSet.has(stackPosKey)) {
        // Found a stack outside the region
        return true;
      }
    }

    // No stacks outside the region - cannot process
    return false;
  }

  /**
   * Process a single disconnected region
   * Rule Reference: Section 12.2 - Processing steps
   */
  private async processOneDisconnectedRegion(
    region: Territory,
    movingPlayer: number
  ): Promise<void> {
    // 1. Get border markers to collapse
    const borderMarkers = this.boardManager.getBorderMarkerPositions(
      region.spaces,
      this.gameState.board
    );

    // 2. Eliminate all rings within the region (all colors) BEFORE
    //    collapsing spaces. This mirrors the Rust engine's
    //    core_apply_disconnect_region behaviour, where internal
    //    eliminations are computed from the pre-collapse stacks.
    let totalRingsEliminated = 0;
    for (const pos of region.spaces) {
      const stack = this.boardManager.getStack(pos, this.gameState.board);
      if (stack) {
        totalRingsEliminated += stack.stackHeight;
        this.boardManager.removeStack(pos, this.gameState.board);
      }
    }

    // 3. Collapse all spaces in the region to the moving player's color
    for (const pos of region.spaces) {
      this.boardManager.setCollapsedSpace(pos, movingPlayer, this.gameState.board);
    }

    // 4. Collapse all border markers to the moving player's color
    for (const pos of borderMarkers) {
      this.boardManager.setCollapsedSpace(pos, movingPlayer, this.gameState.board);
    }

    // Update player's territory count (region spaces + border markers)
    const totalTerritoryGained = region.spaces.length + borderMarkers.length;
    this.updatePlayerTerritorySpaces(movingPlayer, totalTerritoryGained);

    // 5. Update elimination counts - ALL eliminated rings count toward moving player
    this.gameState.totalRingsEliminated += totalRingsEliminated;
    if (!this.gameState.board.eliminatedRings[movingPlayer]) {
      this.gameState.board.eliminatedRings[movingPlayer] = 0;
    }
    this.gameState.board.eliminatedRings[movingPlayer] += totalRingsEliminated;

    // Update player state
    this.updatePlayerEliminatedRings(movingPlayer, totalRingsEliminated);

    // 6. Mandatory self-elimination (one ring or cap from moving player)
    await this.eliminatePlayerRingOrCapWithChoice(movingPlayer);
  }

  /**
   * Process markers along the movement path
   * Rule Reference: Section 8.3 - Marker Interaction
   */
  private processMarkersAlongPath(from: Position, to: Position, player: number): void {
    // Get all positions along the straight line path
    const path = getPathPositions(from, to);

    // Process each position in the path (excluding start and end)
    for (let i = 1; i < path.length - 1; i++) {
      const pos = path[i];
      const marker = this.boardManager.getMarker(pos, this.gameState.board);

      if (marker !== undefined) {
        if (marker === player) {
          // Own marker: collapse to territory (Section 8.3)
          this.boardManager.collapseMarker(pos, player, this.gameState.board);
        } else {
          // Opponent marker: flip to your color (Section 8.3)
          this.boardManager.flipMarker(pos, player, this.gameState.board);
        }
      }
    }
  }

  /**
   * Advance game through phases according to RingRift rules
   * Rule Reference: Section 4, Section 15.2
   *
   * Phase Flow:
   * 1. ring_placement (optional unless no rings on board)
   * 2. movement (required if able)
   * 3. capture (optional to start, mandatory chaining)
   * 4. line_processing (automatic)
   * 5. territory_processing (automatic)
   * 6. Next player's turn
   */
  private advanceGame(): void {
    const deps: TurnEngineDeps = {
      boardManager: this.boardManager,
      ruleEngine: this.ruleEngine,
    };

    const hooks: TurnEngineHooks = {
      eliminatePlayerRingOrCap: (playerNumber: number) => {
        this.eliminatePlayerRingOrCap(playerNumber);
      },
      endGame: (winner?: number, reason?: string) => this.endGame(winner, reason),
    };

    const turnStateBefore: PerTurnState = {
      hasPlacedThisTurn: this.hasPlacedThisTurn,
      mustMoveFromStackKey: this.mustMoveFromStackKey,
    };

    const turnStateAfter = advanceGameForCurrentPlayer(
      this.gameState,
      turnStateBefore,
      deps,
      hooks
    );

    this.hasPlacedThisTurn = turnStateAfter.hasPlacedThisTurn;
    this.mustMoveFromStackKey = turnStateAfter.mustMoveFromStackKey;
  }

  /**
   * Update internal per-turn placement/movement bookkeeping after a move
   * has been applied. This keeps the must-move origin in sync with the
   * stack that was placed or moved, mirroring the sandbox engine’s
   * behaviour while keeping these details off of GameState.
   */
  private updatePerTurnStateAfterMove(move: Move): void {
    const before: PerTurnState = {
      hasPlacedThisTurn: this.hasPlacedThisTurn,
      mustMoveFromStackKey: this.mustMoveFromStackKey,
    };

    const after = updatePerTurnStateAfterMoveTurn(before, move);

    this.hasPlacedThisTurn = after.hasPlacedThisTurn;
    this.mustMoveFromStackKey = after.mustMoveFromStackKey;
  }

  /**
   * Check if player has any valid capture moves available
   * Rule Reference: Section 10.1
   */
  private hasValidCaptures(playerNumber: number): boolean {
    // Delegate to RuleEngine for capture generation so that the
    // decision to enter the capture phase stays in sync with the
    // actual overtaking_capture semantics. We construct a lightweight
    // view of the current state with phase forced to 'capture' for the
    // specified player and ask RuleEngine for valid moves.
    const tempState: GameState = {
      ...this.gameState,
      currentPlayer: playerNumber,
      currentPhase: 'capture',
    };

    const moves = this.ruleEngine.getValidMoves(tempState);
    return moves.some((m) => m.type === 'overtaking_capture');
  }

  /**
   * Check if player has any valid actions available
   * Rule Reference: Section 4.4
   */
  private hasValidActions(playerNumber: number): boolean {
    return (
      this.hasValidPlacements(playerNumber) ||
      this.hasValidMovements(playerNumber) ||
      this.hasValidCaptures(playerNumber)
    );
  }

  /**
   * Check if player has any valid placement moves
   * Rule Reference: Section 4.1, 6.1-6.3
   */
  private hasValidPlacements(playerNumber: number): boolean {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (!player || player.ringsInHand === 0) {
      return false; // No rings in hand to place
    }

    // Check for any empty, non-collapsed spaces
    // For now, we'll do a simple check - in full implementation would check all positions
    // A player can place if they have rings in hand (placement restrictions like movement validation would be checked in the actual move)
    return true; // Simplified - assumes there's usually space to place
  }

  /**
   * Check if player has any valid movement moves
   * Rule Reference: Section 8.1, 8.2
   */
  private hasValidMovements(playerNumber: number): boolean {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, playerNumber);

    if (playerStacks.length === 0) {
      return false; // No stacks to move
    }

    // For each player stack, check if it has any valid moves
    for (const stack of playerStacks) {
      const stackHeight = stack.stackHeight;

      // Check all 8 directions (or 6 for hexagonal)
      const directions = this.getAllDirections();

      for (const direction of directions) {
        // Check if we can move at least stack height in this direction
        let distance = 0;

        for (let step = 1; step <= stackHeight + 5; step++) {
          const nextPos: Position = {
            x: stack.position.x + direction.x * step,
            y: stack.position.y + direction.y * step,
            ...(direction.z !== undefined && { z: (stack.position.z || 0) + direction.z * step }),
          };

          if (!this.boardManager.isValidPosition(nextPos)) {
            break; // Out of bounds
          }

          // Check if this position is blocked (collapsed space or stack)
          if (this.boardManager.isCollapsedSpace(nextPos, this.gameState.board)) {
            break; // Blocked by collapsed space
          }

          const stackAtPos = this.boardManager.getStack(nextPos, this.gameState.board);
          if (stackAtPos) {
            break; // Blocked by another stack
          }

          // This position is reachable
          distance = step;

          // If we've met the minimum distance requirement, we have a valid move
          if (distance >= stackHeight) {
            return true;
          }
        }
      }
    }

    return false; // No valid movements found
  }

  /**
   * Force player to eliminate a cap when blocked with no valid moves
   * Rule Reference: Section 4.4 - Forced Elimination When Blocked
   */
  private processForcedElimination(playerNumber: number): void {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, playerNumber);

    if (playerStacks.length === 0) {
      // No stacks to eliminate from - player forfeits turn
      return;
    }

    // TODO: In full implementation, player should choose which stack
    // For now, eliminate from first stack with a valid cap
    for (const stack of playerStacks) {
      if (stack.capHeight > 0) {
        // Found a stack with a cap, eliminate it
        this.eliminatePlayerRingOrCap(playerNumber);
        return;
      }
    }
  }

  /**
   * Get all movement directions based on board type
   */
  private getAllDirections(): { x: number; y: number; z?: number }[] {
    return getMovementDirectionsForBoardType(this.gameState.boardType);
  }

  /**
   * Get adjacent positions for a given position
   * Uses Moore adjacency (8-direction) for square boards, hexagonal for hex
   */
  private getAdjacentPositions(pos: Position): Position[] {
    const adjacent: Position[] = [];
    const config = BOARD_CONFIGS[this.gameState.boardType];

    if (config.type === 'hexagonal') {
      // Hexagonal adjacency (6 directions)
      const directions = [
        { x: 1, y: 0, z: -1 },
        { x: 0, y: 1, z: -1 },
        { x: -1, y: 1, z: 0 },
        { x: -1, y: 0, z: 1 },
        { x: 0, y: -1, z: 1 },
        { x: 1, y: -1, z: 0 },
      ];

      for (const dir of directions) {
        const newPos: Position = {
          x: pos.x + dir.x,
          y: pos.y + dir.y,
          z: (pos.z || 0) + dir.z,
        };
        if (this.boardManager.isValidPosition(newPos)) {
          adjacent.push(newPos);
        }
      }
    } else {
      // Moore adjacency for square boards (8 directions)
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;

          const newPos: Position = {
            x: pos.x + dx,
            y: pos.y + dy,
          };
          if (this.boardManager.isValidPosition(newPos)) {
            adjacent.push(newPos);
          }
        }
      }
    }

    return adjacent;
  }

  private nextPlayer(): void {
    const currentIndex = this.gameState.players.findIndex(
      (p) => p.playerNumber === this.gameState.currentPlayer
    );
    const nextIndex = (currentIndex + 1) % this.gameState.players.length;
    this.gameState.currentPlayer = this.gameState.players[nextIndex].playerNumber;
  }

  private startPlayerTimer(playerNumber: number): void {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (!player || player.type === 'ai') return;

    const timer = setTimeout(() => {
      // Time expired, forfeit the game
      this.forfeitGame(playerNumber.toString());
    }, player.timeRemaining);

    this.moveTimers.set(playerNumber, timer);
  }

  private stopPlayerTimer(playerNumber: number): void {
    const timer = this.moveTimers.get(playerNumber);
    if (timer) {
      clearTimeout(timer);
      this.moveTimers.delete(playerNumber);
    }
  }

  private endGame(
    winner?: number,
    reason?: string
  ): {
    success: boolean;
    gameResult: GameResult;
  } {
    this.gameState.gameStatus = 'completed';
    this.gameState.winner = winner;

    // Clear all timers
    for (const timer of this.moveTimers.values()) {
      clearTimeout(timer);
    }
    this.moveTimers.clear();

    // Calculate final scores
    const finalScore: { [playerNumber: number]: number } = {};
    for (const player of this.gameState.players) {
      const playerStacks = this.boardManager.getPlayerStacks(
        this.gameState.board,
        player.playerNumber
      );
      const stackCount = playerStacks.reduce((sum, stack) => sum + stack.stackHeight, 0);

      const territories = this.boardManager.findPlayerTerritories(
        this.gameState.board,
        player.playerNumber
      );
      const territorySize = territories.reduce(
        (sum, territory) => sum + territory.spaces.length,
        0
      );

      finalScore[player.playerNumber] = stackCount + territorySize;
    }

    const gameResult: GameResult = {
      ...(winner !== undefined && { winner }),
      reason: (reason as any) || 'game_completed',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: finalScore,
      },
    };

    // Update player ratings if this is a rated game
    if (this.gameState.isRated) {
      this.updatePlayerRatings(gameResult);
    }

    return {
      success: true,
      gameResult,
    };
  }

  private updatePlayerRatings(gameResult: GameResult): void {
    // Rating calculation logic would go here
    const winnerPlayer = this.gameState.players.find((p) => p.playerNumber === gameResult.winner);
    const loserPlayers = this.gameState.players.filter((p) => p.playerNumber !== gameResult.winner);

    // For now, just log the rating update
    console.log('Rating update needed for:', {
      winner: winnerPlayer?.username,
      losers: loserPlayers.map((p) => p.username),
    });
  }

  addSpectator(userId: string): boolean {
    if (!this.gameState.spectators.includes(userId)) {
      this.gameState.spectators.push(userId);
      return true;
    }
    return false;
  }

  removeSpectator(userId: string): boolean {
    const index = this.gameState.spectators.indexOf(userId);
    if (index !== -1) {
      this.gameState.spectators.splice(index, 1);
      return true;
    }
    return false;
  }

  pauseGame(): boolean {
    if (this.gameState.gameStatus === 'active') {
      this.gameState.gameStatus = 'paused';

      // Stop current player's timer
      this.stopPlayerTimer(this.gameState.currentPlayer);

      return true;
    }
    return false;
  }

  resumeGame(): boolean {
    if (this.gameState.gameStatus === 'paused') {
      this.gameState.gameStatus = 'active';

      // Restart current player's timer
      this.startPlayerTimer(this.gameState.currentPlayer);

      return true;
    }
    return false;
  }

  forfeitGame(playerNumber: string): {
    success: boolean;
    gameResult?: GameResult;
  } {
    const winner = this.gameState.players.find(
      (p) => p.playerNumber !== parseInt(playerNumber)
    )?.playerNumber;

    return this.endGame(winner, 'resignation');
  }

  getValidMoves(_playerNumber: number): Move[] {
    const playerNumber = _playerNumber;

    // Only generate moves for the active player to keep server/UI
    // expectations clear.
    if (playerNumber !== this.gameState.currentPlayer) {
      return [];
    }

    // Base move generation comes from RuleEngine, which is responsible
    // for phase-specific legality (placement vs movement vs capture).
    let moves = this.ruleEngine.getValidMoves(this.gameState);

    // When a placement has occurred this turn, restrict movement/capture
    // options so that only the placed/updated stack may move.
    if (
      this.mustMoveFromStackKey &&
      (this.gameState.currentPhase === 'movement' || this.gameState.currentPhase === 'capture')
    ) {
      moves = moves.filter((m) => {
        const isMovementOrCaptureType =
          m.type === 'move_stack' ||
          m.type === 'move_ring' ||
          m.type === 'build_stack' ||
          m.type === 'overtaking_capture';

        if (!isMovementOrCaptureType) {
          // Non-movement moves (e.g. future skip_placement) are not
          // constrained here.
          return true;
        }

        if (!m.from) {
          return false;
        }

        const fromKey = positionToString(m.from);
        return fromKey === this.mustMoveFromStackKey;
      });
    }

    return moves;
  }

  /**
   * Test-only helper: resolve a late-detected "blocked with no moves"
   * situation for the current player.
   *
   * In normal play, TurnEngine is responsible for ensuring that we
   * never enter an interactive phase (ring_placement, movement,
   * capture) for a player who has no legal placement, movement, or
   * capture available. If a test harness nonetheless observes
   * `gameStatus === 'active'` and `getValidMoves(currentPlayer).length
   * === 0` in an interactive phase, it may call this safety net to:
   *
   *   1. Apply forced elimination for any player who controls stacks
   *      but has no legal actions (Section 4.4 / compact rules 2.3),
   *   2. Skip over players who have no material at all,
   *   3. Stop once we either reach a player with at least one legal
   *      action (placement or movement/capture) or the game ends.
   *
   * This helper is intentionally conservative and only used from tests;
   * it does not create any new kinds of actions, it just applies the
   * same forced-elimination / skip semantics the TurnEngine would have
   * applied earlier if the blocked state had been detected on time.
   */
  public resolveBlockedStateForCurrentPlayerForTesting(): void {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    const phase = this.gameState.currentPhase;
    if (phase !== 'ring_placement' && phase !== 'movement' && phase !== 'capture') {
      // Only resolve for interactive phases. Automatic bookkeeping
      // phases should be handled via stepAutomaticPhasesForTesting.
      return;
    }

    const players = this.gameState.players;
    const playerCount = players.length;

    // Compute an upper bound on how many forced-elimination steps we
    // might ever need: the total number of rings still in play (on the
    // board or in hand). Every forced elimination reduces this by at
    // least one, and S is globally non-decreasing, so this bound keeps
    // the resolver from looping forever even in badly broken states.
    const totalRingsRemaining = (() => {
      let total = 0;
      for (const stack of this.gameState.board.stacks.values()) {
        total += stack.stackHeight;
      }
      for (const p of players) {
        total += p.ringsInHand;
      }
      return total;
    })();

    const maxIterations = totalRingsRemaining + playerCount * 4;
    let iterations = 0;

    while (this.gameState.gameStatus === 'active' && iterations < maxIterations) {
      iterations++;

      // 1. First, look for ANY player who has at least one legal
      // placement, movement, or capture under the real rules. If we
      // find such a player, hand the turn to them and reseed the phase
      // so normal play can resume from that point.
      for (const player of players) {
        const playerNumber = player.playerNumber;

        const hasAnyPlacement = (() => {
          if (player.ringsInHand <= 0) {
            return false;
          }

          const tempPlacementState: GameState = {
            ...this.gameState,
            currentPlayer: playerNumber,
            currentPhase: 'ring_placement',
          };

          const placementMoves = this.ruleEngine.getValidMoves(tempPlacementState);
          return placementMoves.some((m) => m.type === 'place_ring');
        })();

        const { hasMovement, hasCapture } = (() => {
          const tempMovementState: GameState = {
            ...this.gameState,
            currentPlayer: playerNumber,
            currentPhase: 'movement',
          };

          const movementMoves = this.ruleEngine.getValidMoves(tempMovementState);
          const hasMovementLocal = movementMoves.some(
            (m) => m.type === 'move_stack' || m.type === 'move_ring' || m.type === 'build_stack'
          );

          const tempCaptureState: GameState = {
            ...this.gameState,
            currentPlayer: playerNumber,
            currentPhase: 'capture',
          };

          const captureMoves = this.ruleEngine.getValidMoves(tempCaptureState);
          const hasCaptureLocal = captureMoves.some((m) => m.type === 'overtaking_capture');

          return { hasMovement: hasMovementLocal, hasCapture: hasCaptureLocal };
        })();

        if (hasAnyPlacement || hasMovement || hasCapture) {
          // Hand the turn to this player and reseed the phase according
          // to whether they still have rings in hand and which kinds of
          // actions are actually available, mirroring the TurnEngine
          // phase flow:
          //   - Prefer ring_placement when placements are legal;
          //   - Otherwise prefer movement when non-capture moves exist;
          //   - Otherwise enter capture phase when only captures remain.
          this.gameState.currentPlayer = playerNumber;

          if (hasAnyPlacement && player.ringsInHand > 0) {
            this.gameState.currentPhase = 'ring_placement';
          } else if (hasMovement) {
            this.gameState.currentPhase = 'movement';
          } else {
            this.gameState.currentPhase = 'capture';
          }

          // Clear per-turn bookkeeping so the next explicit move is
          // treated as the start of a fresh turn.
          this.hasPlacedThisTurn = false;
          this.mustMoveFromStackKey = undefined;
          return;
        }
      }

      // 2. No player has any legal placement/movement/capture. If there
      // are no stacks left on the board, structural terminality has
      // been reached. At this point the compact rules define a
      // stalemate ladder where any rings remaining in hand are treated
      // as eliminated for tie-break purposes (hand → E).
      if (this.gameState.board.stacks.size === 0) {
        let handEliminations = 0;

        for (const player of players) {
          if (player.ringsInHand > 0) {
            const delta = player.ringsInHand;
            player.ringsInHand = 0;
            player.eliminatedRings += delta;
            handEliminations += delta;

            if (!this.gameState.board.eliminatedRings[player.playerNumber]) {
              this.gameState.board.eliminatedRings[player.playerNumber] = 0;
            }
            this.gameState.board.eliminatedRings[player.playerNumber] += delta;
          }
        }

        if (handEliminations > 0) {
          this.gameState.totalRingsEliminated += handEliminations;
        }

        const endCheck = this.ruleEngine.checkGameEnd(this.gameState);
        if (endCheck.isGameOver) {
          this.endGame(endCheck.winner, endCheck.reason || 'structural_stalemate');
        }
        return;
      }

      // 3. Global forced elimination pass: walk players in turn order
      // starting from the current player and eliminate a cap from the
      // first player who still controls stacks. This mirrors the rules
      // text: when everyone is globally blocked but stacks remain,
      // successive forced eliminations must eventually resolve the
      // stalemate until no stacks are left.
      const currentIndex = players.findIndex(
        (p) => p.playerNumber === this.gameState.currentPlayer
      );
      let eliminatedThisIteration = false;

      for (let offset = 0; offset < playerCount; offset++) {
        const idx = (currentIndex + offset) % playerCount;
        const playerNumber = players[idx].playerNumber;
        const stacksForPlayer = this.boardManager.getPlayerStacks(
          this.gameState.board,
          playerNumber
        );

        if (stacksForPlayer.length === 0) {
          continue;
        }

        this.eliminatePlayerRingOrCap(playerNumber);
        eliminatedThisIteration = true;

        const endCheck = this.ruleEngine.checkGameEnd(this.gameState);
        if (endCheck.isGameOver) {
          this.endGame(endCheck.winner, endCheck.reason || 'forced_elimination');
          return;
        }

        // After a forced elimination, make that player the current
        // player in movement phase with fresh per-turn state. The next
        // loop iteration will re-check for available actions for all
        // players from this new board state.
        this.gameState.currentPlayer = playerNumber;
        this.gameState.currentPhase = 'movement';
        this.hasPlacedThisTurn = false;
        this.mustMoveFromStackKey = undefined;
        break;
      }

      if (!eliminatedThisIteration) {
        // Safety: if we somehow failed to eliminate any rings even
        // though stacks remain, bail out rather than spin forever. The
        // caller will continue to treat this as a diagnostic failure.
        return;
      }
    }

    // If we exit because maxIterations was reached while the game is
    // still active, leave the state as-is. The AI simulation harness
    // will continue to treat this as a non-terminating diagnostic
    // failure, but we have avoided an infinite loop inside the engine.
  }

  /**
   * Test-only helper: advance through automatic phases (line_processing
   * and territory_processing) without requiring an explicit player move.
   *
   * This is used by AI simulation/diagnostic harnesses so they do not
   * treat these internal bookkeeping phases as "no legal move" stalls.
   */
  public stepAutomaticPhasesForTesting(): void {
    while (
      this.gameState.gameStatus === 'active' &&
      (this.gameState.currentPhase === 'line_processing' ||
        this.gameState.currentPhase === 'territory_processing')
    ) {
      this.advanceGame();
    }
  }
}
