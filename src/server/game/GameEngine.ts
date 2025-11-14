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
  positionToString
} from '../../shared/types/game';
import { BoardManager } from './BoardManager';
import { RuleEngine } from './RuleEngine';
import { PlayerInteractionManager } from './PlayerInteractionManager';

/**
 * Internal state for enforcing mandatory chain captures during the capture phase.
 *
 * This is intentionally kept out of the wire-level GameState so we can evolve
 * the representation without breaking clients. It is roughly modeled after the
 * Rust engine's `ChainCaptureState` and is used only inside GameEngine.
 */
interface TsChainCaptureSegment {
  from: Position;
  target: Position;
  landing: Position;
  capturedCapHeight: number;
}

interface TsChainCaptureState {
  playerNumber: number;
  startPosition: Position;
  currentPosition: Position;
  segments: TsChainCaptureSegment[];
  // Full capture moves (from=currentPosition) that the player may choose from
  availableMoves: Move[];
  // Positions visited by the capturing stack to help avoid pathological cycles
  visitedPositions: Set<string>;
}

// Timer functions for Node.js environment
declare const setTimeout: (callback: () => void, ms: number) => any;

declare const clearTimeout: (timer: any) => void;

// Using a simple UUID generator for now
function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
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
        isReady: p.type === 'ai' // AI players are always ready
      })),
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      timeControl,
      spectators: [],
      gameStatus: 'waiting',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated,
      maxPlayers: players.length,
      totalRingsInPlay: config.ringsPerPlayer * players.length,
      totalRingsEliminated: 0,
      victoryThreshold: Math.floor(config.ringsPerPlayer * players.length / 2) + 1,
      territoryVictoryThreshold: Math.floor(config.totalSpaces / 2) + 1
    };
  }

  getGameState(): GameState {
    return { ...this.gameState };
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
    const allReady = this.gameState.players.every(p => p.isReady);
    if (!allReady) {
      return false;
    }

    this.gameState.gameStatus = 'active';
    this.gameState.lastMoveAt = new Date();
    
    // Start the first player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);
    
    return true;
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
          error: 'Chain capture in progress: only the capturing player may move'
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
          error: 'Chain capture in progress: must continue capturing with the same stack'
        };
      }
    }

    // Validate the move at the rules level
    const fullMove: Move = {
      ...move,
      id: generateUUID(),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: this.gameState.moveHistory.length + 1
    };

    const validation = this.ruleEngine.validateMove(fullMove, this.gameState);
    if (!validation) {
      return {
        success: false,
        error: 'Invalid move'
      };
    }

    // Capture context needed for chain state bookkeeping (cap height, etc.)
    let capturedCapHeight = 0;
    if (fullMove.type === 'overtaking_capture' && fullMove.captureTarget) {
      const targetStack = this.boardManager.getStack(fullMove.captureTarget, this.gameState.board);
      capturedCapHeight = targetStack ? targetStack.capHeight : 0;
    }

    // Stop current player's timer while we process the move
    this.stopPlayerTimer(this.gameState.currentPlayer);

    // Apply the move to the board state
    const moveResult = this.applyMove(fullMove);

    // Add move to history
    this.gameState.moveHistory.push(fullMove);
    this.gameState.lastMoveAt = new Date();

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
          moveNumber: this.gameState.moveHistory.length + 1
        };

        const _internalResult = this.applyMove(internalMove);
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
    await this.processAutomaticConsequences(moveResult);


    // Check for game end conditions
    const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
    if (gameEndCheck.isGameOver) {
      return this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'unknown');
    }

    // Advance to next phase/player
    this.advanceGame();

    // Start next player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);

    return {
      success: true,
      gameState: this.getGameState()
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
      lineCollapses: [] as LineInfo[]
    };

    switch (move.type) {
      case 'place_ring':
        if (move.to) {
          const newStack: RingStack = {
            position: move.to,
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: move.player,
            rings: [move.player]
          };
          this.boardManager.setStack(move.to, newStack, this.gameState.board);
          
          // Update player state: decrement rings in hand
          const player = this.gameState.players.find(p => p.playerNumber === move.player);
          if (player && player.ringsInHand > 0) {
            player.ringsInHand--;
          }
        }
        break;

      case 'move_ring':
        if (move.from && move.to) {
          const stack = this.boardManager.getStack(move.from, this.gameState.board);
          if (stack) {
            // Rule Reference: Section 4.2.1 - Leave marker on departure space
            this.boardManager.setMarker(move.from, move.player, this.gameState.board);
            
            // Process markers along movement path (Section 8.3)
            this.processMarkersAlongPath(move.from, move.to, move.player);
            
            // Check if landing on same-color marker (Section 8.2)
            const landingMarker = this.boardManager.getMarker(move.to, this.gameState.board);
            if (landingMarker === move.player) {
              this.boardManager.removeMarker(move.to, this.gameState.board);
            }
            
            // Remove stack from source
            this.boardManager.removeStack(move.from, this.gameState.board);
            
            // Normal movement (no capture at landing position)
            const movedStack: RingStack = {
              ...stack,
              position: move.to
            };
            this.boardManager.setStack(move.to, movedStack, this.gameState.board);
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
              rings: remainingRings
            };
            
            const newTargetStack: RingStack = {
              ...targetStack,
              stackHeight: targetStack.stackHeight + move.buildAmount,
              capHeight: Math.max(targetStack.capHeight, move.buildAmount),
              rings: [...targetStack.rings, ...transferRings]
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
    if (!move.from || !move.captureTarget || !move.to) {
      return;
    }

    const segment: TsChainCaptureSegment = {
      from: move.from,
      target: move.captureTarget,
      landing: move.to,
      capturedCapHeight
    };

    if (!this.chainCaptureState) {
      this.chainCaptureState = {
        playerNumber: move.player,
        startPosition: move.from,
        currentPosition: move.to,
        segments: [segment],
        availableMoves: [],
        visitedPositions: new Set<string>([positionToString(move.from)])
      };
      return;
    }

    // Continuing an existing chain
    this.chainCaptureState.currentPosition = move.to;
    this.chainCaptureState.segments.push(segment);
    this.chainCaptureState.visitedPositions.add(positionToString(move.from));
  }

  /**
   * Enumerate all valid capture moves from a given position for the
   * specified player, using the RuleEngine's move generator.
   *
   * This is the TS analogue of the Rust CaptureProcessor's logic for
   * computing follow-up chain options.
   */
  private getCaptureOptionsFromPosition(position: Position, playerNumber: number): Move[] {
    // RuleEngine.getValidMoves will, in capture phase, return only
    // overtaking_capture moves. We further filter by origin position.
    const allMoves = this.ruleEngine.getValidMoves(this.gameState);
    const positionKey = positionToString(position);

    return allMoves.filter(m =>
      m.type === 'overtaking_capture' &&
      m.player === playerNumber &&
      m.from &&
      positionToString(m.from) === positionKey
    );
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
    const state = this.chainCaptureState;
    if (!state) return undefined;

    const options = state.availableMoves;
    if (options.length === 0) {
      return undefined;
    }

    // If there is no interaction manager or only one option, keep
    // behaviour simple for now and just return the sole available move.
    if (!this.interactionManager || options.length === 1) {
      return options[0];
    }

    const interaction = this.requireInteractionManager();

    const choice = {
      id: generateUUID(),
      gameId: this.gameState.id,
      playerNumber: state.playerNumber,
      type: 'capture_direction' as const,
      prompt: 'Choose capture direction and landing position',
      options: options.map(opt => ({
        targetPosition: opt.captureTarget!,
        landingPosition: opt.to,
        // At this point in the chain, the target stack still exists
        // on the board; use its cap height as the capturedCapHeight.
        capturedCapHeight:
          this.boardManager.getStack(opt.captureTarget!, this.gameState.board)?.capHeight || 0
      }))
    };

    const response = await interaction.requestChoice(choice as any);
    const selected = response.selectedOption as {
      targetPosition: Position;
      landingPosition: Position;
      capturedCapHeight: number;
    };

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

    // Check if landing on same-color marker
    const landingMarker = this.boardManager.getMarker(landing, this.gameState.board);
    if (landingMarker === player) {
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
        capHeight: this.calculateCapHeight(remainingTargetRings),
        controllingPlayer: remainingTargetRings[0]
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
      capHeight: this.calculateCapHeight(newRings),
      controllingPlayer: newRings[0]
    };
    this.boardManager.setStack(landing, newStack, this.gameState.board);
  }

  /**
   * Process automatic consequences after a move
   * Rule Reference: Section 4.5 - Post-Movement Processing
   */
  private async processAutomaticConsequences(moveResult: {
    captures: Position[];
    territoryChanges: Territory[];
    lineCollapses: LineInfo[];
  }): Promise<void> {
    // Captures are already processed in applyMove
    
    // Process line formations (Section 11.2, 11.3)
    await this.processLineFormations();
    
    // Process territory disconnections (Section 12.2)
    await this.processDisconnectedRegions();
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
      const playerLines = allLines.filter(
        line => line.player === this.gameState.currentPlayer
      );
      if (playerLines.length === 0) break;

      let lineToProcess: LineInfo;

      if (!this.interactionManager || playerLines.length === 1) {
        // No interaction manager wired yet, or only one choice: keep current behaviour
        lineToProcess = playerLines[0];
      } else {
        const interaction = this.requireInteractionManager();

        const choice = {
          id: generateUUID(),
          gameId: this.gameState.id,
          playerNumber: this.gameState.currentPlayer,
          type: 'line_order' as const,
          prompt: 'Choose which line to process first',
          options: playerLines.map((line, index) => ({
            lineId: String(index),
            markerPositions: line.positions
          }))
        };

        const response = await interaction.requestChoice(choice as any);
        const selected = response.selectedOption as {
          lineId: string;
          markerPositions: Position[];
        };
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

      const choice = {
        id: generateUUID(),
        gameId: this.gameState.id,
        playerNumber: this.gameState.currentPlayer,
        type: 'line_reward_option' as const,
        prompt: 'Choose line reward option',
        options: [
          'option_1_collapse_all_and_eliminate',
          'option_2_min_collapse_no_elimination'
        ] as const
      };

      const response = await interaction.requestChoice(choice as any);
      const selected = response.selectedOption as
        | 'option_1_collapse_all_and_eliminate'
        | 'option_2_min_collapse_no_elimination';

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
      const playerState = this.gameState.players.find(p => p.playerNumber === player);
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
    const capHeight = this.calculateCapHeight(stack.rings);
    
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
        capHeight: this.calculateCapHeight(remainingRings),
        controllingPlayer: remainingRings[0]
      };
      this.boardManager.setStack(stack.position, newStack, this.gameState.board);
    } else {
      // Stack is now empty, remove it
      this.boardManager.removeStack(stack.position, this.gameState.board);
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
      const playerState = this.gameState.players.find(p => p.playerNumber === player);
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

    const choice = {
      id: generateUUID(),
      gameId: this.gameState.id,
      playerNumber: player,
      type: 'ring_elimination' as const,
      prompt: 'Choose which stack to eliminate from',
      options: playerStacks.map(stack => ({
        stackPosition: stack.position,
        capHeight: stack.capHeight,
        totalHeight: stack.stackHeight
      }))
    };

    const response = await interaction.requestChoice(choice as any);
    const selected = response.selectedOption as {
      stackPosition: Position;
      capHeight: number;
      totalHeight: number;
    };

    const selectedKey = positionToString(selected.stackPosition);
    const chosenStack =
      playerStacks.find(s => positionToString(s.position) === selectedKey) ||
      playerStacks[0];

    this.eliminateFromStack(chosenStack, player);
  }

  /**
   * Update player's eliminatedRings counter
   */
  private updatePlayerEliminatedRings(playerNumber: number, count: number): void {
    const player = this.gameState.players.find(p => p.playerNumber === playerNumber);
    if (player) {
      player.eliminatedRings += count;
    }
  }

  /**
   * Update player's territorySpaces counter
   */
  private updatePlayerTerritorySpaces(playerNumber: number, count: number): void {
    const player = this.gameState.players.find(p => p.playerNumber === playerNumber);
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
      
      let region: Territory;
      
      if (!this.interactionManager || disconnectedRegions.length === 1) {
        // No manager or only one region: keep existing behaviour
        region = disconnectedRegions[0];
      } else {
        const interaction = this.requireInteractionManager();
        const choice = {
          id: generateUUID(),
          gameId: this.gameState.id,
          playerNumber: movingPlayer,
          type: 'region_order' as const,
          prompt: 'Choose which disconnected region to process first',
          options: disconnectedRegions.map((r, index) => ({
            regionId: String(index),
            size: r.spaces.length,
            representativePosition: r.spaces[0]
          }))
        };

        const response = await interaction.requestChoice(choice as any);
        const selected = response.selectedOption as {
          regionId: string;
          size: number;
          representativePosition: Position;
        };
        const index = parseInt(selected.regionId, 10);
        region = disconnectedRegions[index] ?? disconnectedRegions[0];
      }
      
      // Self-elimination prerequisite check
      if (!this.canProcessDisconnectedRegion(region, movingPlayer)) {
        // Cannot process this region, skip it
        // In reality, if we can't process any regions, we should break
        // For now, just break to avoid infinite loop
        break;
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
    const regionPositionSet = new Set(region.spaces.map(pos => positionToString(pos)));
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
  private async processOneDisconnectedRegion(region: Territory, movingPlayer: number): Promise<void> {
    // 1. Get border markers to collapse
    const borderMarkers = this.boardManager.getBorderMarkerPositions(
      region.spaces,
      this.gameState.board
    );
    
    // 2. Collapse all spaces in the region to moving player's color
    for (const pos of region.spaces) {
      this.boardManager.setCollapsedSpace(pos, movingPlayer, this.gameState.board);
    }
    
    // 3. Collapse all border markers to moving player's color
    for (const pos of borderMarkers) {
      this.boardManager.setCollapsedSpace(pos, movingPlayer, this.gameState.board);
    }
    
    // Update player's territory count (region spaces + border markers)
    const totalTerritoryGained = region.spaces.length + borderMarkers.length;
    this.updatePlayerTerritorySpaces(movingPlayer, totalTerritoryGained);
    
    // 4. Eliminate all rings within the region (all colors)
    let totalRingsEliminated = 0;
    for (const pos of region.spaces) {
      const stack = this.boardManager.getStack(pos, this.gameState.board);
      if (stack) {
        // Eliminate all rings in this stack
        totalRingsEliminated += stack.stackHeight;
        this.boardManager.removeStack(pos, this.gameState.board);
      }
    }
    
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
   * Calculate cap height for a ring stack
   * Rule Reference: Section 5.2 - Cap height is consecutive rings of same color from top
   */
  private calculateCapHeight(rings: number[]): number {
    if (rings.length === 0) return 0;
    
    const topColor = rings[0];
    let capHeight = 1;
    
    for (let i = 1; i < rings.length; i++) {
      if (rings[i] === topColor) {
        capHeight++;
      } else {
        break;
      }
    }
    
    return capHeight;
  }

  /**
   * Process markers along the movement path
   * Rule Reference: Section 8.3 - Marker Interaction
   */
  private processMarkersAlongPath(from: Position, to: Position, player: number): void {
    // Get all positions along the straight line path
    const path = this.getPathPositions(from, to);
    
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
   * Get all positions along a straight line path
   */
  private getPathPositions(from: Position, to: Position): Position[] {
    const path: Position[] = [from];
    
    // Calculate direction
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = (to.z || 0) - (from.z || 0);
    
    // Normalize to step size of 1
    const steps = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
    const stepX = steps > 0 ? dx / steps : 0;
    const stepY = steps > 0 ? dy / steps : 0;
    const stepZ = steps > 0 ? dz / steps : 0;
    
    // Generate all positions along the path
    for (let i = 1; i <= steps; i++) {
      const pos: Position = {
        x: Math.round(from.x + stepX * i),
        y: Math.round(from.y + stepY * i)
      };
      if (to.z !== undefined) {
        pos.z = Math.round((from.z || 0) + stepZ * i);
      }
      path.push(pos);
    }
    
    return path;
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
    switch (this.gameState.currentPhase) {
      case 'ring_placement':
        // After placing a ring (or skipping), must move
        // Rule Reference: Section 4.1, 4.2
        this.gameState.currentPhase = 'movement';
        break;

      case 'movement':
        // After movement, check if captures are available
        // Rule Reference: Section 4.3
        const canCapture = this.hasValidCaptures(this.gameState.currentPlayer);
        if (canCapture) {
          this.gameState.currentPhase = 'capture';
        } else {
          // Skip to line processing
          this.gameState.currentPhase = 'line_processing';
        }
        break;

      case 'capture':
        // After captures complete, proceed to line processing
        // Rule Reference: Section 4.3, 4.5
        this.gameState.currentPhase = 'line_processing';
        break;

      case 'line_processing':
        // After processing lines, proceed to territory processing
        // Rule Reference: Section 4.5
        this.gameState.currentPhase = 'territory_processing';
        break;

      case 'territory_processing':
        // After processing territory, turn is complete
        // Check if player still has rings/stacks or needs to place
        // Rule Reference: Section 4, Section 4.1
        this.nextPlayer();
        
        // Determine starting phase for next player
        const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, this.gameState.currentPlayer);
        const currentPlayer = this.gameState.players.find(p => p.playerNumber === this.gameState.currentPlayer);
        
        // Rule Reference: Section 4.4 - Forced Elimination When Blocked
        // Check if player has no valid actions but controls stacks
        if (playerStacks.length > 0 && !this.hasValidActions(this.gameState.currentPlayer)) {
          // Player is blocked with stacks - must eliminate a cap
          this.processForcedElimination(this.gameState.currentPlayer);
          
          // After forced elimination, check victory conditions
          const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
          if (gameEndCheck.isGameOver) {
            // Game ended due to forced elimination
            this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'forced_elimination');
            return; // Exit early - game is over
          }
          
          // Continue to next player after forced elimination
          this.nextPlayer();
          
          // Re-evaluate starting phase for the actual next player
          const nextPlayerStacks = this.boardManager.getPlayerStacks(this.gameState.board, this.gameState.currentPlayer);
          const nextPlayer = this.gameState.players.find(p => p.playerNumber === this.gameState.currentPlayer);
          
          if (nextPlayerStacks.length === 0 && nextPlayer && nextPlayer.ringsInHand > 0) {
            this.gameState.currentPhase = 'ring_placement';
          } else if (nextPlayer && nextPlayer.ringsInHand > 0) {
            this.gameState.currentPhase = 'ring_placement';
          } else {
            this.gameState.currentPhase = 'movement';
          }
        } else {
          // Normal turn progression
          if (playerStacks.length === 0 && currentPlayer && currentPlayer.ringsInHand > 0) {
            // No rings on board but has rings in hand - must place
            this.gameState.currentPhase = 'ring_placement';
          } else if (currentPlayer && currentPlayer.ringsInHand > 0) {
            // Has rings in hand and on board - can optionally place
            this.gameState.currentPhase = 'ring_placement';
          } else {
            // No rings in hand or all rings placed - go directly to movement
            this.gameState.currentPhase = 'movement';
          }
        }
        break;
    }
  }

  /**
   * Check if player has any valid capture moves available
   * Rule Reference: Section 10.1
   */
  private hasValidCaptures(playerNumber: number): boolean {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, playerNumber);
    
    for (const stack of playerStacks) {
      // Check all adjacent positions for valid captures
      const adjacentPositions = this.getAdjacentPositions(stack.position);
      for (const adjPos of adjacentPositions) {
        const targetStack = this.boardManager.getStack(adjPos, this.gameState.board);
        if (targetStack && 
            targetStack.controllingPlayer !== playerNumber &&
            stack.capHeight >= targetStack.capHeight) {
          return true; // Found at least one valid capture
        }
      }
    }
    
    return false;
  }

  /**
   * Check if player has any valid placement moves
   * Rule Reference: Section 4.1, 6.1-6.3
   */
  private hasValidPlacements(playerNumber: number): boolean {
    const player = this.gameState.players.find(p => p.playerNumber === playerNumber);
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
        let currentPos = stack.position;
        let distance = 0;
        let pathClear = true;
        
        for (let step = 1; step <= stackHeight + 5; step++) {
          const nextPos: Position = {
            x: stack.position.x + direction.x * step,
            y: stack.position.y + direction.y * step,
            ...(direction.z !== undefined && { z: (stack.position.z || 0) + direction.z * step })
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
   * Get all movement directions based on board type
   */
  private getAllDirections(): { x: number; y: number; z?: number }[] {
    const config = BOARD_CONFIGS[this.gameState.boardType];
    
    if (config.type === 'hexagonal') {
      // Hexagonal directions (6 directions)
      return [
        { x: 1, y: 0, z: -1 },
        { x: 0, y: 1, z: -1 },
        { x: -1, y: 1, z: 0 },
        { x: -1, y: 0, z: 1 },
        { x: 0, y: -1, z: 1 },
        { x: 1, y: -1, z: 0 }
      ];
    } else {
      // Moore adjacency (8 directions) for square boards
      return [
        { x: 1, y: 0 },   // E
        { x: 1, y: 1 },   // SE
        { x: 0, y: 1 },   // S
        { x: -1, y: 1 },  // SW
        { x: -1, y: 0 },  // W
        { x: -1, y: -1 }, // NW
        { x: 0, y: -1 },  // N
        { x: 1, y: -1 }   // NE
      ];
    }
  }

  /**
   * Check if player has any valid actions available
   * Rule Reference: Section 4.4
   */
  private hasValidActions(playerNumber: number): boolean {
    return this.hasValidPlacements(playerNumber) || 
           this.hasValidMovements(playerNumber) || 
           this.hasValidCaptures(playerNumber);
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
        { x: 1, y: -1, z: 0 }
      ];
      
      for (const dir of directions) {
        const newPos: Position = {
          x: pos.x + dir.x,
          y: pos.y + dir.y,
          z: (pos.z || 0) + dir.z
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
            y: pos.y + dy
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
    const currentIndex = this.gameState.players.findIndex(p => p.playerNumber === this.gameState.currentPlayer);
    const nextIndex = (currentIndex + 1) % this.gameState.players.length;
    this.gameState.currentPlayer = this.gameState.players[nextIndex].playerNumber;
  }

  private startPlayerTimer(playerNumber: number): void {
    const player = this.gameState.players.find(p => p.playerNumber === playerNumber);
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

  private endGame(winner?: number, reason?: string): {
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
      const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player.playerNumber);
      const stackCount = playerStacks.reduce((sum, stack) => sum + stack.stackHeight, 0);
      
      const territories = this.boardManager.findPlayerTerritories(this.gameState.board, player.playerNumber);
      const territorySize = territories.reduce((sum, territory) => sum + territory.spaces.length, 0);
      
      finalScore[player.playerNumber] = stackCount + territorySize;
    }

    const gameResult: GameResult = {
      ...(winner !== undefined && { winner }),
      reason: (reason as any) || 'game_completed',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: finalScore
      },
    };

    // Update player ratings if this is a rated game
    if (this.gameState.isRated) {
      this.updatePlayerRatings(gameResult);
    }

    return {
      success: true,
      gameResult
    };
  }

  private updatePlayerRatings(gameResult: GameResult): void {
    // Rating calculation logic would go here
    const winnerPlayer = this.gameState.players.find(p => p.playerNumber === gameResult.winner);
    const loserPlayers = this.gameState.players.filter(p => p.playerNumber !== gameResult.winner);

    // For now, just log the rating update
    console.log('Rating update needed for:', {
      winner: winnerPlayer?.username,
      losers: loserPlayers.map(p => p.username)
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
    const winner = this.gameState.players.find(p => p.playerNumber !== parseInt(playerNumber))?.playerNumber;
    
    return this.endGame(winner, 'resignation');
  }

  getValidMoves(_playerNumber: number): Move[] {
    // This would return all valid moves for the current player
    // For now, return empty array
    return [];
  }

}
