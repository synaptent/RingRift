import type {
  GameState,
  Move,
  Position,
  BoardState,
  RingStack,
  Territory,
  CaptureSegmentBoardView,
  MovementBoardView,
  CaptureBoardAdapters,
  PlacementAggregateContext,
} from '../../shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  positionsEqual,
  calculateCapHeight,
  calculateDistance,
  getPathPositions,
  validateCaptureSegmentOnBoard,
  hasAnyLegalMoveOrCaptureFromOnBoard,
  evaluateVictory,
  enumerateCaptureMoves,
  enumerateSimpleMoveTargetsFromStack,
  canProcessTerritoryRegion,
  enumerateProcessTerritoryRegionMoves,
  enumerateTerritoryEliminationMoves,
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
  findLinesForPlayer,
  // Canonical placement validation + enumeration (TS SSOT)
  validatePlacementOnBoardAggregate,
  enumeratePlacementPositions,
  evaluateSkipPlacementEligibilityAggregate,
  // Chain capture continuation enumeration
  getChainCaptureContinuationInfo,
  // Type guards for move narrowing
  isCaptureMove,
  // Recovery action (RR-CANON-R110–R115)
  isEligibleForRecovery,
  validateRecoverySlide as validateRecoverySlideAggregate,
  enumerateRecoverySlideTargets,
} from '../../shared/engine';
import { getMovementDirectionsForBoardType } from '../../shared/engine/core';
import { validateMoveWithFSM } from '../../shared/engine/fsm/FSMAdapter';
import { flagEnabled, debugLog } from '../../shared/utils/envFlags';
import { BoardManager } from './BoardManager';

/**
 * Backend `RuleEngine` adapter around the shared TS rules engine.
 *
 * This class is allowed to compose shared helpers and aggregates but must
 * not define independent rules semantics. When in doubt, change the shared
 * engine under `src/shared/engine/**` and update this adapter to call into
 * it, following the ownership documented in
 * `docs/RULES_ENGINE_SURFACE_AUDIT.md` (§0 Rules Entry Surfaces).
 */

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
   * Validates a move according to RingRift rules.
   *
   * @deprecated Use `validateMoveWithFSM` from `../../shared/engine/fsm/FSMAdapter`
   * for canonical FSM-based validation per RR-CANON-R070. This method is maintained
   * for backward compatibility but FSM validation is the authoritative validator.
   */
  validateMove(move: Move, gameState: GameState): boolean {
    // Use FSM validation as the canonical source (RR-CANON-R070)
    const fsmResult = validateMoveWithFSM(gameState, move);
    if (!fsmResult.valid) {
      return false;
    }

    // Legacy detailed validation kept for additional checks
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
      case 'recovery_slide':
        // Recovery slide (RR-CANON-R110–R115): marker slide that completes a
        // line, allowing temporarily eliminated players to remain active.
        return this.validateRecoverySlide(move, gameState);
      case 'skip_recovery':
        // RR-CANON-R115: Recovery-eligible player may explicitly skip recovery.
        // Detailed eligibility is enforced by the shared helper.
        return (
          gameState.currentPhase === 'movement' && isEligibleForRecovery(gameState, move.player)
        );
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
   * Backend-specific tightening:
   * - Reject skip_placement when the active player has no rings in hand,
   *   even though the shared aggregate helper treats that case as
   *   semantically fine. When ringsInHand === 0 the TurnEngine is
   *   responsible for advancing phases without requiring an explicit
   *   bookkeeping move.
   *
   * Rule Reference: RR‑CANON‑R080 (optional placement) and related notes in
   * SHARED_ENGINE_CONSOLIDATION_PLAN §Placement Canonical Semantics.
   */
  private validateSkipPlacement(move: Move, gameState: GameState): boolean {
    const playerState = gameState.players.find((p) => p.playerNumber === move.player);
    if (!playerState || playerState.ringsInHand <= 0) {
      return false;
    }

    const eligibility = evaluateSkipPlacementEligibilityAggregate(gameState, move.player);
    return eligibility.eligible;
  }

  /**
   * Validates ring placement according to RingRift rules.
   *
   * This delegates capacity, invariant, and no-dead-placement checks to the
   * shared {@link validatePlacementOnBoardAggregate} helper so that the
   * backend RuleEngine, shared GameEngine, orchestrator, and sandbox all
   * share a single source of truth for placement semantics:
   *
   * - Board geometry / collapsed-space / marker-stack exclusivity.
   * - Per-player ring cap + ringsInHand supply.
   * - Single vs multi-ring placement rules (stacks vs empty cells).
   * - No-dead-placement via hasAnyLegalMoveOrCaptureFromOnBoard.
   *
   * Rule Reference: RR‑CANON‑R070–R072, R080–R082 – placement legality and
   * no-dead-placement.
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

    const ctx: PlacementAggregateContext = {
      boardType,
      player: move.player,
      ringsInHand: playerState.ringsInHand,
      ringsPerPlayerCap: boardConfig.ringsPerPlayer,
    };

    const result = validatePlacementOnBoardAggregate(gameState.board, move.to, requestedCount, ctx);
    return result.valid;
  }

  /**
   * Validates stack movement according to RingRift rules.
   *
   * This now delegates the geometric and path/landing checks to the shared
   * {@link enumerateSimpleMoveTargetsFromStack} helper so that backend
   * semantics stay aligned with the sandbox and shared core rules:
   *
   * - Straight-line movement along canonical directions only.
   * - Distance at least equal to stack height.
   * - Path clear of stacks and collapsed spaces.
   * - Landing on empty spaces or any marker (own or opponent).
   * - Landing on any marker incurs cap-elimination cost per RR-CANON-R091/R092.
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
      this.boardType,
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
   * Validates a recovery slide move according to RR-CANON-R110–R115.
   *
   * A recovery slide is legal when:
   * - The game is in movement phase
   * - The player is eligible for recovery (no stacks, no rings in hand,
   *   has markers, has buried rings)
   * - The slide from→to completes a line of at least lineLength markers
   * - The player has sufficient buried rings to pay the cost (1 for
   *   exact-length or Option 1; 0 for Option 2 on overlength)
   *
   * Delegates to the shared RecoveryAggregate for canonical validation.
   */
  private validateRecoverySlide(move: Move, gameState: GameState): boolean {
    // Recovery slides are only allowed during movement phase
    if (gameState.currentPhase !== 'movement') {
      return false;
    }

    if (!move.from) {
      return false;
    }

    // Delegate to shared aggregate for canonical validation
    // Build the RecoverySlideMove object conditionally to handle exactOptionalPropertyTypes
    const recoveryMove: Parameters<typeof validateRecoverySlideAggregate>[1] = {
      ...move,
      type: 'recovery_slide' as const,
      from: move.from,
      to: move.to,
      // extractionStacks is required but can be derived during apply if not provided
      extractionStacks: [],
    };

    // Only set option if defined (exactOptionalPropertyTypes compliance)
    if (move.recoveryOption !== undefined) {
      recoveryMove.option = move.recoveryOption;
    }

    const result = validateRecoverySlideAggregate(gameState, recoveryMove);
    return result.valid;
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

    return validateCaptureSegmentOnBoard(this.boardType, from, target, landing, player, view);
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
        // RR-CANON-R204 / compact rules §2.1: When ringsInHand == 0 (placement forbidden)
        // but the player controls stacks, enumerate movement moves instead.
        const playerObj = gameState.players.find((p) => p.playerNumber === currentPlayer);
        if (playerObj && playerObj.ringsInHand === 0) {
          // Check if player has any controlled stacks
          let hasControlledStack = false;
          for (const stack of gameState.board.stacks.values()) {
            if (stack.controllingPlayer === currentPlayer && stack.stackHeight > 0) {
              hasControlledStack = true;
              break;
            }
          }
          if (hasControlledStack) {
            // Return movement moves instead of placement/skip
            moves.push(...this.getValidStackMovements(currentPlayer, gameState));
            moves.push(...this.getValidCaptures(currentPlayer, gameState));
            // If no movements or captures available, emit no_placement_action
            if (moves.length === 0) {
              moves.push({
                id: `no_placement_action-${gameState.moveHistory.length + 1}`,
                type: 'no_placement_action',
                player: currentPlayer,
                to: { x: 0, y: 0 },
                timestamp: new Date(),
                thinkTime: 0,
                moveNumber: gameState.moveHistory.length + 1,
              });
            }
            break;
          }
          // No stacks and no rings in hand - emit no_placement_action per RR-CANON-R075
          moves.push({
            id: `no_placement_action-${gameState.moveHistory.length + 1}`,
            type: 'no_placement_action',
            player: currentPlayer,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: gameState.moveHistory.length + 1,
          });
          break;
        }

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

        // If still no moves (no placements, no skip eligible), emit no_placement_action
        // per RR-CANON-R075 to record that the phase was visited.
        if (moves.length === 0) {
          moves.push({
            id: `no_placement_action-${gameState.moveHistory.length + 1}`,
            type: 'no_placement_action',
            player: currentPlayer,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: gameState.moveHistory.length + 1,
          });
        }
        break;
      }
      case 'movement': {
        // During movement phase, expose both simple movements and
        // overtaking capture options so that an initial capture can be
        // chosen directly when legal.
        moves.push(...this.getValidStackMovements(currentPlayer, gameState));
        moves.push(...this.getValidCaptures(currentPlayer, gameState));
        // Recovery slides (RR-CANON-R110–R115): marker slides that complete
        // lines for temporarily eliminated players.
        moves.push(...this.getValidRecoverySlides(currentPlayer, gameState));
        // If no movements, captures, or recovery available, emit no_movement_action
        // per RR-CANON-R075 to record that the phase was visited.
        if (moves.length === 0) {
          moves.push({
            id: `no_movement_action-${gameState.moveHistory.length + 1}`,
            type: 'no_movement_action',
            player: currentPlayer,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: gameState.moveHistory.length + 1,
          });
        }
        break;
      }
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
      case 'chain_capture': {
        // Enumerate chain capture continuations using chainCapturePosition from GameState.
        // This allows AIEngine and other callers to get valid moves without needing
        // access to GameEngine's internal chainCaptureState.
        const chainPos = gameState.chainCapturePosition;
        if (chainPos) {
          const info = getChainCaptureContinuationInfo(
            gameState,
            gameState.currentPlayer,
            chainPos
          );
          if (info.mustContinue) {
            moves.push(...info.availableContinuations);
          }
        }
        break;
      }
      case 'forced_elimination': {
        // In the forced_elimination phase (7th phase per RR-CANON-R070),
        // the player must eliminate from one of their controlled stacks.
        // Uses the same elimination decision moves as territory processing.
        moves.push(...this.getValidEliminationDecisionMoves(gameState));
        break;
      }
    }

    return moves;
  }

  /**
   * Gets valid ring placement moves for the given player using the shared,
   * canonical PlacementAggregate:
   *
   * - Capacity (per-player ring cap + ringsInHand supply).
   * - Board geometry / collapsed-space / marker-stack exclusivity.
   * - Single vs multi-ring placement rules.
   * - No-dead-placement via hasAnyLegalMoveOrCaptureFromOnBoard.
   *
   * Enumeration first delegates to {@link enumeratePlacementPositions} to
   * discover all cells where at least one ring placement is legal, then
   * uses {@link validatePlacementOnBoardAggregate} to derive the per-move
   * placementCount surface (1..3 on empty cells, 1 on existing stacks)
   * without re-encoding cap or ray-walk semantics locally.
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

    // Canonical enumeration of all cells where at least one ring placement
    // is legal for this player under the current state (including
    // no-dead-placement).
    const legalPositions = enumeratePlacementPositions(gameState, player);
    if (legalPositions.length === 0) {
      return moves;
    }

    const baseMoveNumber = gameState.moveHistory.length + 1;

    for (const pos of legalPositions) {
      const posKey = positionToString(pos);
      const stack = board.stacks.get(posKey);
      const isOccupied = !!(stack && stack.rings.length > 0);

      // Per-cell cap is 1 when placing onto an existing stack and up to 3
      // on empty cells; global capacity and supply constraints are enforced
      // by the shared validator.
      const perCellCap = isOccupied ? 1 : 3;

      for (let count = 1; count <= perCellCap; count++) {
        const ctx: PlacementAggregateContext = {
          boardType,
          player,
          ringsInHand: playerState.ringsInHand,
          ringsPerPlayerCap: boardConfig.ringsPerPlayer,
        };

        const validation = validatePlacementOnBoardAggregate(board, pos, count, ctx);
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
      const targets = enumerateSimpleMoveTargetsFromStack(this.boardType, stackPos, player, view);

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

    const TRACE_DEBUG_ENABLED = flagEnabled('RINGRIFT_TRACE_DEBUG');

    const baseMoveNumber = gameState.moveHistory.length + 1;
    const moves: Move[] = [];

    for (const stackPos of playerStacks) {
      const fromKey = positionToString(stackPos);
      const attackerStack = board.stacks.get(fromKey);
      if (!attackerStack || attackerStack.controllingPlayer !== player) {
        continue;
      }

      const rawMoves = enumerateCaptureMoves(
        this.boardType,
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
        debugLog(TRACE_DEBUG_ENABLED, '[RuleEngine.getValidCaptures debug seed17]', {
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

      // Filter to typed capture moves for type-safe field access
      rawMoves.filter(isCaptureMove).forEach((m, index) => {
        moves.push({
          ...m,
          id:
            m.id && m.id.length > 0
              ? m.id
              : `capture-${positionToString(m.from)}-${positionToString(
                  m.captureTarget
                )}-${positionToString(m.to)}-${index}`,
          moveNumber: baseMoveNumber,
        });
      });
    }

    return moves;
  }

  /**
   * Gets valid recovery slide moves for the given player using the shared
   * RecoveryAggregate (RR-CANON-R110–R115).
   *
   * Recovery slides are only available when the player is "temporarily
   * eliminated" (no stacks, no rings in hand, has markers, has buried rings)
   * and can slide a marker to complete a line.
   */
  private getValidRecoverySlides(player: number, gameState: GameState): Move[] {
    // Early exit if player is not eligible for recovery
    if (!isEligibleForRecovery(gameState, player)) {
      return [];
    }

    const baseMoveNumber = gameState.moveHistory.length + 1;
    const targets = enumerateRecoverySlideTargets(gameState, player);
    const moves: Move[] = [];

    for (const target of targets) {
      // Create base move for this slide target
      const baseMove: Move = {
        id: `recovery-${positionToString(target.from)}-${positionToString(target.to)}`,
        type: 'recovery_slide',
        player,
        from: target.from,
        to: target.to,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: baseMoveNumber,
      };

      // For overlength lines, create separate moves for Option 1 and Option 2
      if (target.isOverlength && target.option2Available) {
        // Option 1 (collapse all, cost 1) - always available for overlength
        const opt1Move: Move = {
          ...baseMove,
          id: `recovery-${positionToString(target.from)}-${positionToString(target.to)}-opt1`,
          recoveryOption: 1,
        };
        moves.push(opt1Move);

        // Option 2 (collapse minimum, free)
        const opt2Move: Move = {
          ...baseMove,
          id: `recovery-${positionToString(target.from)}-${positionToString(target.to)}-opt2`,
          recoveryOption: 2,
        };
        moves.push(opt2Move);
      } else {
        // Exact-length: only Option 1 semantics (costs 1 buried ring)
        moves.push(baseMove);
      }
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
    return getMovementDirectionsForBoardType(this.boardType);
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
    return calculateDistance(this.boardType, from, to);
  }

  private areAdjacent(pos1: Position, pos2: Position): boolean {
    const distance = this.calculateDistance(pos1, pos2);
    return distance === 1;
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

    return hasAnyLegalMoveOrCaptureFromOnBoard(this.boardType, from, player, view);
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

    // Keep diagnostics-only helpers and type adapters referenced so that
    // ts-node/TypeScript with noUnusedLocals can compile backend entrypoints
    // (including orchestrator soak harnesses) without treating them as dead code.
    void this.canProcessDisconnectedRegionForRules;
    void this.hasAnyLegalMoveOrCaptureFrom;

    const _debugRingStack: RingStack | null = null;
    void _debugRingStack;
  }
}
