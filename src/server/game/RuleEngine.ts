import {
  GameState,
  Move,
  Position,
  BoardState,
  RingStack,
  positionToString,
  positionsEqual,
  BOARD_CONFIGS,
  Territory,
} from '../../shared/types/game';
import { BoardManager } from './BoardManager';
import {
  calculateCapHeight,
  calculateDistance,
  getMovementDirectionsForBoardType,
  getPathPositions,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView,
  MovementBoardView,
  hasAnyLegalMoveOrCaptureFromOnBoard,
  applyMarkerEffectsAlongPathOnBoard,
  MarkerPathHelpers,
} from '../../shared/engine/core';
import { evaluateVictory } from '../../shared/engine/victoryLogic';
import { enumerateCaptureMoves, CaptureBoardAdapters } from '../../shared/engine/captureLogic';
import { enumerateSimpleMoveTargetsFromStack } from '../../shared/engine/movementLogic';
import {
  filterProcessableTerritoryRegions,
  canProcessTerritoryRegion,
} from '../../shared/engine/territoryProcessing';
import {
  enumerateProcessTerritoryRegionMoves,
  enumerateTerritoryEliminationMoves,
} from '../../shared/engine/territoryDecisionHelpers';
import {
  validatePlacementOnBoard,
  PlacementContext,
} from '../../shared/engine/validators/PlacementValidator';
import {
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
} from '../../shared/engine/lineDecisionHelpers';
import { findLinesForPlayer } from '../../shared/engine/lineDetection';

export class RuleEngine {
  private boardManager: BoardManager;
  private boardConfig: (typeof BOARD_CONFIGS)[keyof typeof BOARD_CONFIGS];
  private boardType: keyof typeof BOARD_CONFIGS;

  constructor(boardManager: BoardManager, boardType: keyof typeof BOARD_CONFIGS) {
    this.boardManager = boardManager;
    this.boardConfig = BOARD_CONFIGS[boardType];
    this.boardType = boardType;

    // Keep selected internal helpers referenced so ts-node/TypeScript with
    // noUnusedLocals enabled does not treat them as dead code. This has
    // no runtime effect; it only preserves helpers for diagnostics and
    // future rule-engine extensions.
    this._debugUseInternalHelpers();
  }

  /**
   * Validates a move according to RingRift rules
   */
  validateMove(move: Move, gameState: GameState): boolean {
    // Basic validation
    if (!this.isValidPlayer(move.player, gameState)) {
      return false;
    }

    if (!this.isPlayerTurn(move.player, gameState)) {
      return false;
    }

    // Validate based on move type and game phase
    switch (move.type) {
      case 'place_ring':
        return this.validateRingPlacement(move, gameState);
      case 'move_ring': // legacy alias for simple stack movement
      case 'move_stack':
        return this.validateStackMovement(move, gameState);
      case 'overtaking_capture':
        return this.validateCapture(move, gameState);
      case 'continue_capture_segment':
        return this.validateChainCaptureContinuation(move, gameState);
      case 'process_line':
      case 'choose_line_reward':
        return this.validateLineProcessingMove(move, gameState);
      case 'process_territory_region':
        return this.validateTerritoryProcessingMove(move, gameState);
      case 'eliminate_rings_from_stack':
        return this.validateEliminationMove(move, gameState);
      case 'skip_placement':
        // Trivial validation: skipping placement is only allowed during the
        // ring_placement phase, and only when placement is optional under
        // the rules (i.e. the player has rings in hand *and* at least one
        // controlled stack with a legal move or capture available).
        return this.validateSkipPlacement(move, gameState);
      default:
        return false;
    }
  }

  /**
   * Validates a skip_placement move. This is a no-op move that is only
   * legal during the ring_placement phase when placement is *optional*:
   * the player has rings in hand, controls at least one stack on the
   * board, and has at least one legal move or capture from some
   * controlled stack. It is *not* allowed to skip when placement is
   * mandatory (no stacks on board, or stacks with no legal moves).
   *
   * Rule Reference: Section 4.1 / 2.1 – optional placement when
   * movement/capture is available.
   */
  private validateSkipPlacement(move: Move, gameState: GameState): boolean {
    if (gameState.currentPhase !== 'ring_placement') {
      return false;
    }

    const playerState = gameState.players.find((p) => p.playerNumber === move.player);
    if (!playerState || playerState.ringsInHand <= 0) {
      // No rings to optionally place – skipping is meaningless here.
      return false;
    }

    const board = gameState.board;
    const playerStacks = this.getPlayerStacks(move.player, board);

    // If the player has no stacks at all, placement is mandatory.
    if (playerStacks.length === 0) {
      return false;
    }

    // Check if at least one controlled stack has a legal move or capture
    // in the *current* board state. If none do, placement is mandatory
    // (the player is effectively blocked without placing).
    const hasAnyAction = playerStacks.some((pos) =>
      this.hasAnyLegalMoveOrCaptureFrom(pos, move.player, board)
    );

    return hasAnyAction;
  }

  /**
   * Validates ring placement according to RingRift rules.
   *
   * This now delegates the detailed capacity, invariant, and no-dead-placement
   * checks to the shared {@link validatePlacementOnBoard} helper so that the
   * backend RuleEngine, shared GameEngine, and sandbox all share a single
   * source of truth for placement semantics:
   *
   * - Board geometry / collapsed-space / marker-stack exclusivity.
   * - Per-player ring cap + ringsInHand supply.
   * - Single vs multi-ring placement rules (stacks vs empty cells).
   * - No-dead-placement via hasAnyLegalMoveOrCaptureFromOnBoard.
   *
   * Rule Reference: Section 7.1 – Placement must leave at least one legal move
   * or capture from the resulting stack.
   */
  private validateRingPlacement(move: Move, gameState: GameState): boolean {
    // Ring placement is only allowed during the ring_placement phase.
    if (gameState.currentPhase !== 'ring_placement') {
      return false;
    }

    const playerState = gameState.players.find((p) => p.playerNumber === move.player);
    if (!playerState || playerState.ringsInHand <= 0) {
      return false;
    }

    const boardType = gameState.board.type;
    const boardConfig = BOARD_CONFIGS[boardType];
    const requestedCount = move.placementCount ?? 1;

    const ctx: PlacementContext = {
      boardType,
      player: move.player,
      ringsInHand: playerState.ringsInHand,
      ringsPerPlayerCap: boardConfig.ringsPerPlayer,
    };

    const result = validatePlacementOnBoard(gameState.board, move.to, requestedCount, ctx);
    return result.valid;
  }

  /**
   * Validates stack movement according to RingRift rules.
   *
   * This now delegates the geometric and path/landing checks to the shared
   * {@link enumerateSimpleMoveTargetsFromStack} helper so that backend
   * semantics stay aligned with the sandbox and shared GameEngine:
   *
   * - Straight-line movement along canonical directions only.
   * - Distance at least equal to stack height.
   * - Path clear of stacks and collapsed spaces.
   * - Landing on empty spaces, own markers, or any stack (merge).
   * - Landing on opponent markers remains illegal.
   *
   * Rule Reference: Section 8.2, FAQ Q2.
   */
  private validateStackMovement(move: Move, gameState: GameState): boolean {
    // Stack movement is allowed during movement or capture phases.
    if (gameState.currentPhase !== 'movement' && gameState.currentPhase !== 'capture') {
      return false;
    }

    if (!move.from) {
      return false;
    }

    const fromKey = positionToString(move.from);
    const sourceStack = gameState.board.stacks.get(fromKey);
    if (!sourceStack || sourceStack.controllingPlayer !== move.player) {
      return false;
    }

    // Build a MovementBoardView over the current board so we can reuse the
    // shared reachability helper for both validation and enumeration.
    const view: MovementBoardView = {
      isValidPosition: (pos: Position) => this.boardManager.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => this.boardManager.isCollapsedSpace(pos, gameState.board),
      getStackAt: (pos: Position) => {
        const key = positionToString(pos);
        const stack = gameState.board.stacks.get(key);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight,
        };
      },
      getMarkerOwner: (pos: Position) => this.boardManager.getMarker(pos, gameState.board),
    };

    const targets = enumerateSimpleMoveTargetsFromStack(
      this.boardConfig.type as any,
      move.from,
      move.player,
      view
    );

    // A simple move is legal iff the requested destination matches one of the
    // shared helper's reachable targets.
    return targets.some((t) => positionsEqual(t.to, move.to));
  }

  /**
   * Checks if the path between two positions is clear
   * Rule Reference: Section 8.1, Section 8.2
   */
  private isPathClear(from: Position, to: Position, board: BoardState): boolean {
    // Get path positions (excluding start and end)
    const pathPositions = getPathPositions(from, to).slice(1, -1);

    // Check each position along the path
    for (const pos of pathPositions) {
      const posKey = positionToString(pos);

      // Cannot pass through collapsed spaces
      if (this.boardManager.isCollapsedSpace(pos, board)) {
        return false;
      }

      // Cannot pass through other rings/stacks
      const stack = board.stacks.get(posKey);
      if (stack && stack.rings.length > 0) {
        return false;
      }

      // Markers are OK to pass through - they get flipped/collapsed
    }

    return true;
  }

  /**
   * Validates overtaking capture move according to RingRift rules
   * Rule Reference: Section 10.1, Section 10.2
   */

  /**
   * Validates overtaking capture move according to RingRift rules
   * Rule Reference: Section 10.1, Section 10.2
   */
  private validateCapture(move: Move, gameState: GameState): boolean {
    // Captures are only allowed during interactive phases (movement/capture)
    // and during the dedicated chain_capture phase used for explicit capture
    // continuation segments. This allows:
    //   - an initial overtaking capture to be chosen as the first action
    //     of the turn from either movement or capture phase, and
    //   - internal enumeration of follow-up segments during chain_capture
    //     while still disallowing captures during placement and
    //     post-processing phases.
    if (
      gameState.currentPhase !== 'capture' &&
      gameState.currentPhase !== 'movement' &&
      gameState.currentPhase !== 'chain_capture'
    ) {
      return false;
    }

    if (!move.from || !move.captureTarget) {
      return false;
    }

    return this.validateCaptureSegment(
      move.from,
      move.captureTarget,
      move.to,
      move.player,
      gameState.board
    );
  }

  /**
   * Validates a follow-up capture segment during the dedicated chain_capture
   * phase. The geometric/path semantics are identical to an overtaking_capture
   * segment; the only difference is that this move is only legal while an
   * existing chain is in progress.
   *
   * The GameEngine is responsible for enforcing that the move's `from`
   * position matches the current chain origin and that the player matches
   * the chain owner; here we only enforce phase and segment-level legality.
   */
  private validateChainCaptureContinuation(move: Move, gameState: GameState): boolean {
    if (gameState.currentPhase !== 'chain_capture') {
      return false;
    }

    if (!move.from || !move.captureTarget) {
      return false;
    }

    return this.validateCaptureSegment(
      move.from,
      move.captureTarget,
      move.to,
      move.player,
      gameState.board
    );
  }

  /**
   * Core validation for a single overtaking capture segment from `from`
   * over `target` to `landing`. This mirrors the Rust engine's
   * `validate_capture_segment` at a high level and is used both by
   * `validateCapture` and by capture move generation.
   */
  private validateCaptureSegment(
    from: Position,
    target: Position,
    landing: Position,
    player: number,
    board: BoardState
  ): boolean {
    const view: CaptureSegmentBoardView = {
      isValidPosition: (pos: Position) => this.boardManager.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => this.boardManager.isCollapsedSpace(pos, board),
      getStackAt: (pos: Position) => {
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight,
        };
      },
      getMarkerOwner: (pos: Position) => this.boardManager.getMarker(pos, board),
    };

    return validateCaptureSegmentOnBoard(
      this.boardType as any,
      from,
      target,
      landing,
      player,
      view
    );
  }

  /**
   * Processes a move and returns the new game state
   */
  processMove(move: Move, gameState: GameState): GameState {
    const newState = this.cloneGameState(gameState);

    switch (move.type) {
      case 'place_ring':
        this.processRingPlacement(move, newState);
        break;
      case 'move_stack':
        this.processStackMovement(move, newState);
        break;
      case 'overtaking_capture':
        this.processCapture(move, newState);
        break;
    }

    // Process automatic consequences
    this.processLineFormation(newState);
    this.processTerritoryDisconnection(newState);

    return newState;
  }

  /**
   * Processes ring placement
   */
  private processRingPlacement(move: Move, gameState: GameState): void {
    const newStack: RingStack = {
      position: move.to,
      rings: [move.player],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: move.player,
    };

    this.boardManager.setStack(move.to, newStack, gameState.board);
  }

  /**
   * Processes stack movement
   */
  private processStackMovement(move: Move, gameState: GameState): void {
    if (!move.from) return;

    const fromKey = positionToString(move.from);
    const toKey = positionToString(move.to);

    const sourceStack = gameState.board.stacks.get(fromKey);
    if (!sourceStack) return;

    const destinationStack = gameState.board.stacks.get(toKey);

    // Apply marker effects along the path (leave departure marker, flip/collapse intermediate)
    const markerHelpers: MarkerPathHelpers = {
      setMarker: (pos, player, b) => this.boardManager.setMarker(pos, player, b),
      collapseMarker: (pos, player, b) => this.boardManager.collapseMarker(pos, player, b),
      flipMarker: (pos, player, b) => this.boardManager.flipMarker(pos, player, b),
    };

    // Check if landing on own marker before applying effects (which might remove it)
    const landingMarker = this.boardManager.getMarker(move.to, gameState.board);
    const landedOnOwnMarker = landingMarker === move.player;

    applyMarkerEffectsAlongPathOnBoard(
      gameState.board,
      move.from,
      move.to,
      move.player,
      markerHelpers
    );

    if (destinationStack && destinationStack.rings.length > 0) {
      // Merge stacks
      const mergedStack: RingStack = {
        position: move.to,
        rings: [...destinationStack.rings, ...sourceStack.rings],
        stackHeight: destinationStack.stackHeight + sourceStack.stackHeight,
        capHeight: sourceStack.capHeight, // Moving stack's cap becomes new cap
        controllingPlayer: sourceStack.controllingPlayer,
      };
      this.boardManager.setStack(move.to, mergedStack, gameState.board);
    } else {
      // Move to empty position (or position that had a marker which is now removed)
      const movedStack: RingStack = {
        ...sourceStack,
        position: move.to,
      };
      this.boardManager.setStack(move.to, movedStack, gameState.board);
    }

    // Remove from source
    this.boardManager.removeStack(move.from, gameState.board);

    // Handle landing on own marker: eliminate top ring
    if (landedOnOwnMarker) {
      const stackAtLanding = gameState.board.stacks.get(toKey);
      if (stackAtLanding && stackAtLanding.stackHeight > 0) {
        const [, ...remainingRings] = stackAtLanding.rings;

        if (remainingRings.length > 0) {
          const newStack: RingStack = {
            ...stackAtLanding,
            rings: remainingRings,
            stackHeight: remainingRings.length,
            capHeight: calculateCapHeight(remainingRings),
            controllingPlayer: remainingRings[0],
          };
          this.boardManager.setStack(move.to, newStack, gameState.board);
        } else {
          this.boardManager.removeStack(move.to, gameState.board);
        }

        // Update elimination counts
        const player = move.player; // The mover is the one who gets credited/penalized?
        // Rules say: "If you land on your own marker... remove the top ring of your stack... This counts as an eliminated ring."
        // It counts towards the player's eliminated rings total.

        gameState.totalRingsEliminated += 1;
        if (!gameState.board.eliminatedRings[player]) {
          gameState.board.eliminatedRings[player] = 0;
        }
        gameState.board.eliminatedRings[player] += 1;

        const playerState = gameState.players.find((p) => p.playerNumber === player);
        if (playerState) {
          playerState.eliminatedRings += 1;
        }
      }
    }
  }

  /**
   * Processes capture with chain reactions
   */
  private processCapture(move: Move, gameState: GameState): void {
    if (!move.from || !move.capturedStacks || !move.captureTarget) return;

    const fromKey = positionToString(move.from);
    const attackerStack = gameState.board.stacks.get(fromKey);
    if (!attackerStack) return;

    const capturedStacks: RingStack[] = move.capturedStacks;

    // Apply marker effects along the path.
    // For capture, we must handle the path from `from` to `captureTarget` (leaving departure marker)
    // and from `captureTarget` to `to` (NOT leaving departure marker on target).

    const markerHelpers: MarkerPathHelpers = {
      setMarker: (pos, player, b) => this.boardManager.setMarker(pos, player, b),
      collapseMarker: (pos, player, b) => this.boardManager.collapseMarker(pos, player, b),
      flipMarker: (pos, player, b) => this.boardManager.flipMarker(pos, player, b),
    };

    // Check if landing on own marker before applying effects
    const landingMarker = this.boardManager.getMarker(move.to, gameState.board);
    const landedOnOwnMarker = landingMarker === move.player;

    // 1. Path from start to target
    applyMarkerEffectsAlongPathOnBoard(
      gameState.board,
      move.from,
      move.captureTarget,
      move.player,
      markerHelpers,
      { leaveDepartureMarker: true }
    );

    // 2. Path from target to landing
    applyMarkerEffectsAlongPathOnBoard(
      gameState.board,
      move.captureTarget,
      move.to,
      move.player,
      markerHelpers,
      { leaveDepartureMarker: false }
    );

    // Apply overtaking one ring at a time, mirroring the GameEngine and
    // Rust behaviour: each capture segment takes only the top ring of the
    // target stack and adds it to the bottom of the attacker. Target
    // stacks that become empty are removed.
    let updatedAttacker = attackerStack;

    for (const capturedStack of capturedStacks) {
      const capturedKey = positionToString(capturedStack.position);
      const currentTarget = gameState.board.stacks.get(capturedKey);
      if (!currentTarget || currentTarget.rings.length === 0) {
        continue;
      }

      const [capturedRing, ...remaining] = currentTarget.rings;

      if (remaining.length > 0) {
        const newTarget: RingStack = {
          ...currentTarget,
          rings: remaining,
          stackHeight: remaining.length,
          capHeight: calculateCapHeight(remaining),
          controllingPlayer: remaining[0],
        };
        this.boardManager.setStack(capturedStack.position, newTarget, gameState.board);
      } else {
        this.boardManager.removeStack(capturedStack.position, gameState.board);
      }

      const newRings = [...updatedAttacker.rings, capturedRing];

      updatedAttacker = {
        ...updatedAttacker,
        rings: newRings,
        stackHeight: newRings.length,
        capHeight: calculateCapHeight(newRings),
        controllingPlayer: newRings[0],
      };
    }

    // Move the updated attacker stack to the destination
    const toKey = positionToString(move.to);
    const movedStack: RingStack = {
      ...updatedAttacker,
      position: move.to,
    };
    this.boardManager.setStack(move.to, movedStack, gameState.board);
    this.boardManager.removeStack(move.from, gameState.board);

    // Handle landing on own marker: eliminate top ring
    if (landedOnOwnMarker) {
      const stackAtLanding = gameState.board.stacks.get(toKey);
      if (stackAtLanding && stackAtLanding.stackHeight > 0) {
        const [, ...remainingRings] = stackAtLanding.rings;

        if (remainingRings.length > 0) {
          const newStack: RingStack = {
            ...stackAtLanding,
            rings: remainingRings,
            stackHeight: remainingRings.length,
            capHeight: calculateCapHeight(remainingRings),
            controllingPlayer: remainingRings[0],
          };
          this.boardManager.setStack(move.to, newStack, gameState.board);
        } else {
          this.boardManager.removeStack(move.to, gameState.board);
        }

        // Update elimination counts
        const player = move.player;
        gameState.totalRingsEliminated += 1;
        if (!gameState.board.eliminatedRings[player]) {
          gameState.board.eliminatedRings[player] = 0;
        }
        gameState.board.eliminatedRings[player] += 1;

        const playerState = gameState.players.find((p) => p.playerNumber === player);
        if (playerState) {
          playerState.eliminatedRings += 1;
        }
      }
    }

    // Check for chain reactions (legacy behaviour; GameEngine now drives
    // chain captures via chainCaptureState and CaptureDirectionChoice).
    this.processChainReactions(move.from, gameState);
  }
  /**
   * Processes chain reactions from captures
   */
  private processChainReactions(triggerPos: Position, gameState: GameState): void {
    const triggerKey = positionToString(triggerPos);
    const triggerStack = gameState.board.stacks.get(triggerKey);
    if (!triggerStack) return;

    const adjacentPositions = this.getAdjacentPositions(triggerPos);

    for (const adjPos of adjacentPositions) {
      const adjKey = positionToString(adjPos);
      const adjStack = gameState.board.stacks.get(adjKey);
      if (
        adjStack &&
        adjStack.controllingPlayer !== triggerStack.controllingPlayer &&
        triggerStack.capHeight >= adjStack.capHeight
      ) {
        // Trigger another capture
        const captureMove: Move = {
          id: `chain-${Date.now()}`,
          type: 'overtaking_capture',
          player: triggerStack.controllingPlayer,
          from: triggerPos,
          to: adjPos,
          capturedStacks: [adjStack],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: gameState.moveHistory.length + 1,
        };

        this.processCapture(captureMove, gameState);
      }
    }
  }

  /**
   * Processes line formation and marker collapse
   */
  private processLineFormation(gameState: GameState): void {
    const lines = this.boardManager.findAllLines(gameState.board);

    for (const line of lines) {
      if (line.positions.length >= this.boardConfig.lineLength) {
        // Collapse line - remove all stacks in the line
        for (const pos of line.positions) {
          this.boardManager.removeStack(pos, gameState.board);
        }
      }
    }
  }

  /**
   * Processes territory disconnection
   */
  private processTerritoryDisconnection(gameState: GameState): void {
    // Check territories for each player
    for (const player of gameState.players) {
      const territories = this.boardManager.findAllTerritories(
        player.playerNumber,
        gameState.board
      );

      for (const territory of territories) {
        if (territory.isDisconnected) {
          // Remove all stacks in disconnected territory
          for (const pos of territory.spaces) {
            this.boardManager.removeStack(pos, gameState.board);
          }
        }
      }
    }
  }

  /**
   * Checks for game end conditions using the shared, canonical victory helper
   * from src/shared/engine/victoryLogic so that backend RuleEngine, shared
   * GameEngine, and sandbox all share a single source of truth for
   * ring-elimination, territory-control, and stalemate ladder semantics.
   */
  checkGameEnd(gameState: GameState): { isGameOver: boolean; winner?: number; reason?: string } {
    const result = evaluateVictory(gameState);

    if (!result.isGameOver) {
      return { isGameOver: false };
    }

    const response: { isGameOver: boolean; winner?: number; reason?: string } = {
      isGameOver: true,
    };

    if (result.winner !== undefined) {
      response.winner = result.winner;
    }
    if (result.reason !== undefined) {
      response.reason = result.reason;
    }

    return response;
  }

  /**
   * Gets valid moves for the current game state
   */
  getValidMoves(gameState: GameState): Move[] {
    const moves: Move[] = [];
    const currentPlayer = gameState.currentPlayer;

    switch (gameState.currentPhase) {
      case 'ring_placement': {
        // Generate all legal ring placements for the active player.
        moves.push(...this.getValidRingPlacements(currentPlayer, gameState));

        // Also expose the skip_placement no-op when (and only when) placement
        // is optional under the rules. This mirrors TurnEngine.hasValidPlacements
        // and prevents "active game with no legal moves" states in
        // ring_placement when the player has rings in hand *and* at least one
        // legal move/capture from a controlled stack.
        const skipMoveCandidate: Move = {
          id: 'skip_placement',
          type: 'skip_placement',
          player: currentPlayer,
          // Skip placement is a pure phase transition with no board
          // coordinates. However, the shared Move type requires a `to`
          // position, so we supply a harmless sentinel value that is
          // never inspected by skip_placement-specific logic.
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: gameState.moveHistory.length + 1,
        };

        if (this.validateSkipPlacement(skipMoveCandidate, gameState)) {
          moves.push(skipMoveCandidate);
        }
        break;
      }
      case 'movement':
        // During movement phase, expose both simple movements and
        // overtaking capture options so that an initial capture can be
        // chosen directly when legal.
        moves.push(...this.getValidStackMovements(currentPlayer, gameState));
        moves.push(...this.getValidCaptures(currentPlayer, gameState));
        break;
      case 'capture':
        moves.push(...this.getValidCaptures(currentPlayer, gameState));
        break;
      case 'line_processing':
        // Enumerate canonical line-processing decision moves (process_line
        // and choose_line_reward) for the current player based on the
        // current board state. This mirrors the GameEngine helper but is
        // stateless and suitable for unit tests that operate directly on
        // RuleEngine and GameState.
        moves.push(...this.getValidLineProcessingDecisionMoves(gameState));
        break;
      case 'territory_processing': {
        // Enumerate canonical territory-processing decision moves
        // (process_territory_region) for the current player, subject to
        // the self-elimination prerequisite from §12.2 / FAQ Q23. Only
        // when no such regions remain do we surface explicit
        // self-elimination decisions via eliminate_rings_from_stack
        // moves.
        const regionMoves = this.getValidTerritoryProcessingDecisionMoves(gameState);
        moves.push(...regionMoves);

        if (regionMoves.length === 0) {
          moves.push(...this.getValidEliminationDecisionMoves(gameState));
        }
        break;
      }
      case 'chain_capture':
        // Advanced-phase enumeration for chain_capture is handled by
        // GameEngine.getValidMoves, which has access to the internal
        // chainCaptureState. RuleEngine remains focused on segment-level
        // validation for overtaking_capture / continue_capture_segment.
        break;
    }

    return moves;
  }

  /**
   * Gets valid ring placement moves for the given player using the shared,
   * canonical placement validator. This keeps backend move enumeration in
   * lock-step with the shared GameEngine and sandbox:
   *
   * - Capacity (per-player ring cap + ringsInHand supply).
   * - Board geometry / collapsed-space / marker-stack exclusivity.
   * - Single vs multi-ring placement rules.
   * - No-dead-placement via hasAnyLegalMoveOrCaptureFromOnBoard.
   */
  private getValidRingPlacements(player: number, gameState: GameState): Move[] {
    const moves: Move[] = [];
    const playerState = gameState.players.find((p) => p.playerNumber === player);
    if (!playerState || playerState.ringsInHand <= 0) {
      return moves;
    }

    const board = gameState.board;
    const boardType = board.type;
    const boardConfig = BOARD_CONFIGS[boardType];
    const allPositions = this.boardManager.getAllPositions();

    // Compute rings on board for capacity checks, mirroring the legacy
    // RuleEngine behaviour (crediting whole stacks to the controlling
    // player). This keeps per-player caps consistent across hosts even
    // though it slightly over-approximates rings for mixed-colour stacks.
    const playerStacks = this.getPlayerStacks(player, board);
    const ringsOnBoard = playerStacks.reduce((sum, pos) => {
      const stackKey = positionToString(pos);
      const stackAtPos = board.stacks.get(stackKey);
      return sum + (stackAtPos ? stackAtPos.rings.length : 0);
    }, 0);

    const remainingByCap = boardConfig.ringsPerPlayer - ringsOnBoard;
    const remainingBySupply = playerState.ringsInHand;
    const maxAvailableGlobal = Math.min(remainingByCap, remainingBySupply);

    if (maxAvailableGlobal <= 0) {
      return moves;
    }

    const baseCtx: PlacementContext = {
      boardType,
      player,
      ringsInHand: playerState.ringsInHand,
      ringsPerPlayerCap: boardConfig.ringsPerPlayer,
      ringsOnBoard,
      maxAvailableGlobal,
    };

    const baseMoveNumber = gameState.moveHistory.length + 1;

    for (const pos of allPositions) {
      const posKey = positionToString(pos);
      const stack = board.stacks.get(posKey);
      const isOccupied = !!(stack && stack.rings.length > 0);

      const maxPerPlacement = isOccupied ? 1 : Math.min(3, maxAvailableGlobal);
      if (maxPerPlacement <= 0) {
        continue;
      }

      for (let count = 1; count <= maxPerPlacement; count++) {
        const validation = validatePlacementOnBoard(board, pos, count, baseCtx);
        if (!validation.valid) {
          continue;
        }

        const candidate: Move = {
          id: isOccupied
            ? `place-${positionToString(pos)}-stack`
            : `place-${positionToString(pos)}-x${count}`,
          type: 'place_ring',
          player,
          to: pos,
          placedOnStack: isOccupied,
          placementCount: count,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: baseMoveNumber,
        };

        moves.push(candidate);
      }
    }

    return moves;
  }

  /**
   * Gets valid stack movement moves.
   *
   * This is now a thin adapter over the shared
   * {@link enumerateSimpleMoveTargetsFromStack} helper, which encodes the
   * canonical non-capturing movement semantics for both backend and sandbox.
   */
  private getValidStackMovements(player: number, gameState: GameState): Move[] {
    const moves: Move[] = [];
    const board = gameState.board;
    const playerStacks = this.getPlayerStacks(player, board);

    if (playerStacks.length === 0) {
      return moves;
    }

    const view: MovementBoardView = {
      isValidPosition: (pos: Position) => this.boardManager.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => this.boardManager.isCollapsedSpace(pos, board),
      getStackAt: (pos: Position) => {
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight,
        };
      },
      getMarkerOwner: (pos: Position) => this.boardManager.getMarker(pos, board),
    };

    const baseMoveNumber = gameState.moveHistory.length + 1;

    for (const stackPos of playerStacks) {
      const targets = enumerateSimpleMoveTargetsFromStack(
        this.boardConfig.type as any,
        stackPos,
        player,
        view
      );

      for (const target of targets) {
        moves.push({
          id: `move-${positionToString(target.from)}-${positionToString(target.to)}`,
          type: 'move_stack',
          player,
          from: target.from,
          to: target.to,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: baseMoveNumber,
        });
      }
    }

    return moves;
  }

  /**
   * Gets valid capture moves using a directional enumeration similar to the
   * Rust CaptureProcessor: from each stack, walk along rays in all movement
   * directions, find the first capturable target on each ray, then enumerate
   * valid landing positions beyond that target.
   *
   * Rule Reference: Section 10.1, 10.2 - Overtaking capture requirements
   */
  private getValidCaptures(player: number, gameState: GameState): Move[] {
    const board = gameState.board;
    const playerStacks = this.getPlayerStacks(player, board);

    if (playerStacks.length === 0) {
      return [];
    }

    // Adapt the current board view to the shared capture enumerator so that
    // backend capture enumeration stays in lock-step with the sandbox and
    // shared core rules.
    const adapters: CaptureBoardAdapters = {
      isValidPosition: (pos: Position) => this.boardManager.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => this.boardManager.isCollapsedSpace(pos, board),
      getStackAt: (pos: Position) => {
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight,
        };
      },
      getMarkerOwner: (pos: Position) => this.boardManager.getMarker(pos, board),
    };

    const TRACE_DEBUG_ENABLED =
      typeof process !== 'undefined' &&
      !!(process as any).env &&
      ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

    const baseMoveNumber = gameState.moveHistory.length + 1;
    const moves: Move[] = [];

    for (const stackPos of playerStacks) {
      const fromKey = positionToString(stackPos);
      const attackerStack = board.stacks.get(fromKey);
      if (!attackerStack || attackerStack.controllingPlayer !== player) {
        continue;
      }

      const rawMoves = enumerateCaptureMoves(
        this.boardConfig.type as any,
        stackPos,
        player,
        adapters,
        baseMoveNumber
      );

      if (
        TRACE_DEBUG_ENABLED &&
        this.boardType === 'square8' &&
        stackPos.x === 2 &&
        stackPos.y === 0 &&
        player === 2
      ) {
        const attackerKey = positionToString(stackPos);
        const attackerDebug = board.stacks.get(attackerKey);
        const targetKey = positionToString({ x: 3, y: 1 } as Position);
        const targetDebug = board.stacks.get(targetKey);
        // eslint-disable-next-line no-console
        console.log('[RuleEngine.getValidCaptures debug seed17]', {
          from: attackerKey,
          player,
          attackerStack: attackerDebug,
          targetKey,
          targetStack: targetDebug,
          rawMoves: rawMoves.map((m) => ({
            type: m.type,
            from: m.from ? positionToString(m.from) : undefined,
            captureTarget: m.captureTarget ? positionToString(m.captureTarget) : undefined,
            to: m.to ? positionToString(m.to) : undefined,
          })),
        });
      }

      rawMoves.forEach((m, index) => {
        moves.push({
          ...m,
          id:
            m.id && m.id.length > 0
              ? m.id
              : `capture-${positionToString(m.from!)}-${positionToString(
                  m.captureTarget!
                )}-${positionToString(m.to!)}-${index}`,
          moveNumber: baseMoveNumber,
        });
      });
    }

    return moves;
  }

  /**
   * Enumerate canonical line-processing decision moves (process_line and
   * choose_line_reward) for the current state.
   *
   * This is now a thin, stateless adapter over the shared
   * {@link enumerateProcessLineMoves} and {@link enumerateChooseLineRewardMoves}
   * helpers so that backend RuleEngine, shared GameEngine, and sandbox all
   * share a single source of truth for:
   *
   * - Which lines exist for the moving player.
   * - How process_line decisions identify those lines (formedLines[0]).
   * - How overlength reward options (collapse-all vs minimum-collapse
   *   contiguous segments) are surfaced as choose_line_reward Moves.
   *
   * Tests that care about the precise reward surface should assert against
   * the shared helper behaviour rather than re-encoding counting logic here.
   */
  private getValidLineProcessingDecisionMoves(gameState: GameState): Move[] {
    if (gameState.currentPhase !== 'line_processing') {
      return [];
    }

    const movingPlayer = gameState.currentPlayer;

    // Base process_line decisions: one per player-owned line, using the
    // canonical lineDecisionHelpers enumeration. We prefer the board's
    // formedLines cache when present since the backend BoardManager keeps
    // it up to date after movement/capture.
    const processMoves = enumerateProcessLineMoves(gameState, movingPlayer, {
      detectionMode: 'use_board_cache',
    });

    // Reward decisions are enumerated per line index. We drive the index
    // sequence via findLinesForPlayer so that enumeration order remains
    // stable even if detection internals change.
    const playerLines = findLinesForPlayer(gameState.board, movingPlayer);
    const rewardMoves: Move[] = [];

    playerLines.forEach((_line, index) => {
      rewardMoves.push(...enumerateChooseLineRewardMoves(gameState, movingPlayer, index));
    });

    return [...processMoves, ...rewardMoves];
  }

  /**
   * Enumerate canonical territory-processing decision moves
   * (process_territory_region) for the current state. This now delegates to
   * the shared {@link enumerateProcessTerritoryRegionMoves} helper so that
   * backend RuleEngine, shared GameEngine, and sandbox all share a single
   * source of truth for:
   *
   * - Disconnected-region detection.
   * - Q23 self-elimination gating (must control a stack outside the region).
   * - Move ID / payload conventions for process_territory_region.
   */
  private getValidTerritoryProcessingDecisionMoves(gameState: GameState): Move[] {
    if (gameState.currentPhase !== 'territory_processing') {
      return [];
    }

    const movingPlayer = gameState.currentPlayer;
    return enumerateProcessTerritoryRegionMoves(gameState, movingPlayer);
  }

  /**
   * Enumerate explicit self-elimination decisions for the current player
   * as 'eliminate_rings_from_stack' Moves during the territory_processing
   * phase. These are only exposed once no eligible disconnected regions
   * remain for the moving player, mirroring the "region first, then
   * self-elimination" ordering from §12.2 / FAQ Q23.
   *
   * This now delegates to the shared {@link enumerateTerritoryEliminationMoves}
   * helper so backend RuleEngine, shared GameEngine, and sandbox enumerate
   * identical elimination options with consistent diagnostics.
   */
  private getValidEliminationDecisionMoves(gameState: GameState): Move[] {
    if (gameState.currentPhase !== 'territory_processing') {
      return [];
    }

    const movingPlayer = gameState.currentPlayer;
    return enumerateTerritoryEliminationMoves(gameState, movingPlayer);
  }

  /**
   * Local self-elimination prerequisite check for RulesEngine-based
   * territory-processing enumeration: the moving player must control at
   * least one stack/cap outside the disconnected region.
   */
  private canProcessDisconnectedRegionForRules(
    gameState: GameState,
    region: Territory,
    player: number
  ): boolean {
    // Thin wrapper around the shared self-elimination prerequisite helper
    // so RulesEngine-based tests can continue to call this method directly
    // while the actual gating semantics remain centralised in
    // src/shared/engine/territoryProcessing.
    return canProcessTerritoryRegion(gameState.board, region, { player });
  }

  /**
   * Directions along which captures may occur, based on board type.
   * For square boards we use the 8 Moore directions; for hex we use
   * the 6 standard cube-coordinate directions.
   */
  private getCaptureDirections(): { x: number; y: number; z?: number }[] {
    // Capture directions mirror movement directions: 8-direction Moore
    // adjacency for square boards and the 6 standard cube directions for hex.
    return getMovementDirectionsForBoardType(this.boardConfig.type as any);
  }

  /**
   * Helper methods
   */
  private isValidPlayer(player: number, gameState: GameState): boolean {
    return gameState.players.some((p) => p.playerNumber === player);
  }

  private isPlayerTurn(player: number, gameState: GameState): boolean {
    return gameState.currentPlayer === player;
  }

  private getPlayerStacks(player: number, board: BoardState): Position[] {
    const positions: Position[] = [];

    for (const [, stack] of board.stacks) {
      if (stack.controllingPlayer === player) {
        positions.push(stack.position);
      }
    }

    return positions;
  }

  private getPlayerStats(gameState: GameState): {
    [player: number]: { totalRings: number; controlledPositions: number };
  } {
    const stats: { [player: number]: { totalRings: number; controlledPositions: number } } = {};

    // Initialize stats
    for (const player of gameState.players) {
      stats[player.playerNumber] = { totalRings: 0, controlledPositions: 0 };
    }

    // Count rings and controlled positions
    for (const [, stack] of gameState.board.stacks) {
      if (stack.rings.length > 0) {
        const player = stack.controllingPlayer;
        stats[player].totalRings += stack.rings.length;
        stats[player].controlledPositions += 1;
      }
    }

    return stats;
  }

  private calculateDistance(from: Position, to: Position): number {
    // Delegate to the shared core helper so distance semantics match
    // ClientSandboxEngine and capture validation (Chebyshev for
    // square boards, cube distance for hex).
    return calculateDistance(this.boardConfig.type as any, from, to);
  }

  private areAdjacent(pos1: Position, pos2: Position): boolean {
    const distance = this.calculateDistance(pos1, pos2);
    return distance === 1;
  }

  private getAdjacentPositions(pos: Position): Position[] {
    const adjacent: Position[] = [];

    if (this.boardConfig.type === 'hexagonal') {
      // Hexagonal adjacency
      const directions = [
        { x: 1, y: 0, z: -1 }, // East
        { x: 0, y: 1, z: -1 }, // Southeast
        { x: -1, y: 1, z: 0 }, // Southwest
        { x: -1, y: 0, z: 1 }, // West
        { x: 0, y: -1, z: 1 }, // Northwest
        { x: 1, y: -1, z: 0 }, // Northeast
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

  /**
   * True if the move from `from` to `to` is along a straight ray consistent
   * with the board's movement directions (8-direction Moore for square,
   * 6 cube-coordinate axes for hex). This mirrors the directional checks
   * used by the shared capture validator and sandbox movement.
   */
  private isStraightLineMovement(from: Position, to: Position): boolean {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = (to.z || 0) - (from.z || 0);

    if (this.boardConfig.type === 'hexagonal') {
      // In cube coordinates, an axis-aligned ray changes exactly two
      // coordinates (the third is implied by x + y + z = 0).
      const coordChanges = [dx !== 0, dy !== 0, dz !== 0].filter(Boolean).length;
      return coordChanges === 2;
    }

    // Square boards: orthogonal or diagonal only.
    if (dx === 0 && dy === 0) {
      return false;
    }
    if (dx !== 0 && dy !== 0 && Math.abs(dx) !== Math.abs(dy)) {
      return false;
    }
    return true;
  }

  private cloneGameState(gameState: GameState): GameState {
    return {
      ...gameState,
      board: {
        ...gameState.board,
        stacks: new Map(gameState.board.stacks),
        markers: new Map(gameState.board.markers),
        territories: new Map(gameState.board.territories),
        formedLines: [...gameState.board.formedLines],
        eliminatedRings: { ...gameState.board.eliminatedRings },
      },
      moveHistory: [...gameState.moveHistory],
      players: [...gameState.players],
      spectators: [...gameState.spectators],
    };
  }

  /**
   * Checks if a stack at the given position has any legal moves or captures.
   * Used for placement validation to ensure rings aren't placed in positions
   * with no legal moves.
   *
   * Rule Reference: Section 7.1 - Must have at least one legal move or capture
   */
  private hasAnyLegalMoveOrCaptureFrom(from: Position, player: number, board: BoardState): boolean {
    const view: MovementBoardView = {
      isValidPosition: (pos: Position) => this.boardManager.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => this.boardManager.isCollapsedSpace(pos, board),
      getStackAt: (pos: Position) => {
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight,
        };
      },
      getMarkerOwner: (pos: Position) => this.boardManager.getMarker(pos, board),
    };

    return hasAnyLegalMoveOrCaptureFromOnBoard(this.boardConfig.type as any, from, player, view);
  }

  /**
   * Helper for hypothetical move checking - simpler path validation
   * that works with a board state rather than game state.
   */
  private isPathClearForHypothetical(from: Position, to: Position, board: BoardState): boolean {
    const pathPositions = getPathPositions(from, to).slice(1, -1);

    for (const pos of pathPositions) {
      if (this.boardManager.isCollapsedSpace(pos, board)) {
        return false;
      }

      const posKey = positionToString(pos);
      const stack = board.stacks.get(posKey);
      if (stack && stack.rings.length > 0) {
        return false;
      }
    }

    return true;
  }

  /**
   * Internal no-op hook to keep selected helper methods referenced so that
   * ts-node/TypeScript with noUnusedLocals can compile the server in dev
   * without treating them as dead code. This has no runtime impact; it
   * simply preserves helpers for parity/debug tooling and future rules
   * extensions.
   */
  /**
   * Validate line-processing decision moves (process_line / choose_line_reward)
   * during the dedicated 'line_processing' phase.
   *
   * For now we treat these as structurally valid when:
   * - The game is in line_processing phase,
   * - The move is made by the current player, and
   * - When formedLines[0] is provided, it matches one of the currently
   *   detected lines for that player according to BoardManager.findAllLines.
   *
   * This keeps decision moves aligned with the same line view used for
   * enumeration in GameEngine.getValidLineProcessingMoves.
   */
  private validateLineProcessingMove(move: Move, gameState: GameState): boolean {
    if (gameState.currentPhase !== 'line_processing') {
      return false;
    }

    if (move.player !== gameState.currentPlayer) {
      return false;
    }

    const allLines = this.boardManager.findAllLines(gameState.board);
    const playerLines = allLines.filter((line) => line.player === move.player);

    if (playerLines.length === 0) {
      return false;
    }

    // When the Move carries a concrete line in formedLines[0], ensure it
    // corresponds to one of the currently-detected lines by comparing
    // ordered marker positions.
    if (move.formedLines && move.formedLines.length > 0) {
      const target = move.formedLines[0];

      const matchesExisting = playerLines.some((line) => {
        if (line.positions.length !== target.positions.length) {
          return false;
        }
        return line.positions.every((pos, idx) => positionsEqual(pos, target.positions[idx]));
      });

      if (!matchesExisting) {
        return false;
      }
    }

    return true;
  }

  /**
   * Validate territory-processing decision moves (process_territory_region)
   * during the dedicated 'territory_processing' phase.
   *
   * A move is considered valid when:
   * - The game is in territory_processing phase,
   * - The move is made by the current player, and
   * - When disconnectedRegions[0] is provided, its space set matches one of
   *   the regions currently reported by BoardManager.findDisconnectedRegions
   *   for the moving player.
   *
   * The stricter self-elimination prerequisite (having an outside stack) is
   * enforced by GameEngine.canProcessDisconnectedRegion when enumerating
   * these moves; here we focus on structural identity.
   */
  private validateTerritoryProcessingMove(move: Move, gameState: GameState): boolean {
    if (gameState.currentPhase !== 'territory_processing') {
      return false;
    }

    if (move.player !== gameState.currentPlayer) {
      return false;
    }

    const disconnectedRegions = this.boardManager.findDisconnectedRegions(
      gameState.board,
      move.player
    );

    if (!disconnectedRegions || disconnectedRegions.length === 0) {
      return false;
    }

    if (move.disconnectedRegions && move.disconnectedRegions.length > 0) {
      const target = move.disconnectedRegions[0];
      const targetKeySet = new Set(target.spaces.map((pos: Position) => positionToString(pos)));

      const matchesExisting = disconnectedRegions.some((region: Territory) => {
        if (region.spaces.length !== target.spaces.length) {
          return false;
        }

        const regionKeySet = new Set(region.spaces.map((pos: Position) => positionToString(pos)));

        if (regionKeySet.size !== targetKeySet.size) {
          return false;
        }

        for (const key of targetKeySet) {
          if (!regionKeySet.has(key)) {
            return false;
          }
        }

        return true;
      });

      if (!matchesExisting) {
        return false;
      }
    }

    return true;
  }

  /**
   * Validate explicit elimination decision moves ('eliminate_rings_from_stack').
   *
   * Currently scoped to the territory_processing phase where the moving
   * player is required to self-eliminate from one of their controlled
   * stacks (or hand) after processing a disconnected region. The Move
   * identifies the target stack via `to`; eliminatedRings, when present,
   * is treated as diagnostic and checked for consistency with the cap
   * geometry.
   */
  private validateEliminationMove(move: Move, gameState: GameState): boolean {
    if (gameState.currentPhase !== 'territory_processing') {
      return false;
    }

    if (move.player !== gameState.currentPlayer) {
      return false;
    }

    if (!move.to) {
      return false;
    }

    const toKey = positionToString(move.to);
    const stack = gameState.board.stacks.get(toKey);
    if (!stack || stack.controllingPlayer !== move.player) {
      return false;
    }

    const capHeight = calculateCapHeight(stack.rings);
    if (capHeight <= 0) {
      return false;
    }

    // Optional diagnostic consistency: when eliminatedRings carries an
    // entry for the moving player, ensure it does not exceed the current
    // cap height and is strictly positive.
    if (move.eliminatedRings && move.eliminatedRings.length > 0) {
      const entry = move.eliminatedRings.find((e) => e.player === move.player);
      if (entry && (entry.count <= 0 || entry.count > capHeight)) {
        return false;
      }
    }

    return true;
  }

  private _debugUseInternalHelpers(): void {
    void this.getPlayerStats;
    void this.areAdjacent;
    void this.isPathClearForHypothetical;
    void this.getCaptureDirections;
    // Also keep legacy path/geometry helpers referenced so TypeScript
    // does not flag them as unused while they remain valuable for
    // diagnostics and future rule-engine extensions.
    void this.isPathClear;
    void this.isStraightLineMovement;
  }
}
