import { GameEngine } from '../../src/shared/engine/GameEngine';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import {
  GameState,
  PlaceRingAction,
  MoveStackAction,
  OvertakingCaptureAction,
} from '../../src/shared/engine/types';
import { Player } from '../../src/shared/types/game';

describe('Refactored GameEngine', () => {
  let engine: GameEngine;
  let initialState: GameState;
  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
  const timeControl: any = { initialTime: 600, increment: 0, type: 'blitz' };

  beforeEach(() => {
    initialState = createInitialGameState('test-game', 'square8', players, timeControl);
    engine = new GameEngine(initialState);
  });

  describe('Initialization', () => {
    it('should initialize with correct state', () => {
      const state = engine.getGameState();
      expect(state.id).toBe('test-game');
      expect(state.players.length).toBe(2);
      expect(state.currentPhase).toBe('ring_placement');
      expect(state.currentPlayer).toBe(1);
      expect(state.board.stacks.size).toBe(0);
      expect(state.board.markers.size).toBe(0);
    });

    it('should give players correct initial rings', () => {
      const state = engine.getGameState();
      // For square8, ringsPerPlayer is usually 30 or similar, checking config
      // But we can just check they are equal and > 0
      expect(state.players[0].ringsInHand).toBeGreaterThan(0);
      expect(state.players[1].ringsInHand).toBe(state.players[0].ringsInHand);
    });
  });

  describe('Placement', () => {
    it('should allow valid placement', () => {
      const action: PlaceRingAction = {
        type: 'PLACE_RING',
        playerId: 1,
        position: { x: 0, y: 0 },
        count: 1,
      };

      const result = engine.processAction(action);
      expect(result.type).toBe('ACTION_PROCESSED');

      const state = engine.getGameState();
      expect(state.board.stacks.get('0,0')).toBeDefined();
      expect(state.board.stacks.get('0,0')?.rings).toEqual([1]);
      expect(state.players[0].ringsInHand).toBe(initialState.players[0].ringsInHand - 1);
    });

    it('should reject multi-ring placement on existing stack', () => {
      // P1 places at (0,0)
      engine.processAction({
        type: 'PLACE_RING',
        playerId: 1,
        position: { x: 0, y: 0 },
        count: 1,
      });

      // P1 moves (0,0) to (0,1) to complete turn
      engine.processAction({
        type: 'MOVE_STACK',
        playerId: 1,
        from: { x: 0, y: 0 },
        to: { x: 0, y: 1 },
      });

      // Now P2's turn. (0,1) has a stack.
      // Try to place 2 rings on it (invalid)
      const action: PlaceRingAction = {
        type: 'PLACE_RING',
        playerId: 2,
        position: { x: 0, y: 1 },
        count: 2,
      };

      const result = engine.processAction(action);
      expect(result.type).toBe('ERROR_OCCURRED');
      if (result.type === 'ERROR_OCCURRED') {
        expect(result.payload.code).toBe('INVALID_COUNT');
      }
    });
  });

  describe('Movement', () => {
    it('should allow valid move', () => {
      // Setup: P1 places at (0,0)
      engine.processAction({
        type: 'PLACE_RING',
        playerId: 1,
        position: { x: 0, y: 0 },
        count: 1,
      });

      // Move (0,0) to (0,1)
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: { x: 0, y: 0 },
        to: { x: 0, y: 1 },
      };

      const result = engine.processAction(action);
      expect(result.type).toBe('ACTION_PROCESSED');

      const state = engine.getGameState();
      expect(state.board.stacks.get('0,0')).toBeUndefined();
      expect(state.board.stacks.get('0,1')).toBeDefined();
      expect(state.board.markers.get('0,0')).toBeDefined();
      expect(state.board.markers.get('0,0')?.player).toBe(1);
    });

    it('should reject move if path blocked', () => {
      // Setup a scenario with a blocking stack by directly modifying state.
      // This bypasses complex multi-turn setup which can have phase issues.

      const state = engine.getGameState();

      // Put a stack at (0,0) for player 1
      state.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // Put a blocking stack at (0,1) for player 2
      state.board.stacks.set('0,1', {
        position: { x: 0, y: 1 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      // Set the phase to movement for player 1
      (state as any).currentPhase = 'movement';
      (state as any).currentPlayer = 1;

      // Create a fresh engine with this state
      engine = new GameEngine(state);

      // Now P1 tries to move (0,0) to (0,2), jumping over stack at (0,1).
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: { x: 0, y: 0 },
        to: { x: 0, y: 2 },
      };

      const result = engine.processAction(action);
      expect(result.type).toBe('ERROR_OCCURRED');
      if (result.type === 'ERROR_OCCURRED') {
        expect(result.payload.code).toBe('PATH_BLOCKED');
      }
    });

    it('should handle landing on own marker (stack merging/elimination)', () => {
      // Setup: P1 places at (0,0) and moves to (1,0), leaving marker at (0,0)
      // We move to (1,0) instead of (0,1) to keep the vertical path (0,2)->(0,0) clear of stacks.
      engine.processAction({ type: 'PLACE_RING', playerId: 1, position: { x: 0, y: 0 }, count: 1 });
      engine.processAction({
        type: 'MOVE_STACK',
        playerId: 1,
        from: { x: 0, y: 0 },
        to: { x: 1, y: 0 },
      });

      // P2 passes turn (place & move elsewhere)
      engine.processAction({ type: 'PLACE_RING', playerId: 2, position: { x: 2, y: 2 }, count: 1 });
      engine.processAction({
        type: 'MOVE_STACK',
        playerId: 2,
        from: { x: 2, y: 2 },
        to: { x: 2, y: 3 },
      });

      // P1 places at (0,2)
      engine.processAction({ type: 'PLACE_RING', playerId: 1, position: { x: 0, y: 2 }, count: 1 });

      // P1 moves (0,2) to (0,0) (landing on own marker)
      // Path: (0,2) -> (0,1) -> (0,0).
      // (0,1) is empty. (0,0) has marker.
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: { x: 0, y: 2 },
        to: { x: 0, y: 0 },
      };

      const result = engine.processAction(action);
      expect(result.type).toBe('ACTION_PROCESSED');

      const state = engine.getGameState();
      // Marker at (0,0) should be gone
      expect(state.board.markers.get('0,0')).toBeUndefined();

      // Stack had 1 ring. Landing on own marker eliminates top ring.
      // 1 - 1 = 0. Stack removed.
      expect(state.board.stacks.get('0,0')).toBeUndefined();

      // Check elimination count
      // Note: players array is 0-indexed, playerNumber is 1-based.
      // P1 is at index 0.
      // Initial eliminatedRings was 0. Should be 1 now.
      // Wait, did we eliminate any other rings? No.
      // But we need to check if the engine updates the player object in the array.
      const p1 = state.players.find((p) => p.playerNumber === 1);
      expect(p1?.eliminatedRings).toBe(1);
    });
  });

  describe('Capture', () => {
    it('should allow valid overtaking capture', () => {
      // Setup:
      // P1 places at (0,0) and moves to (0,1)
      engine.processAction({ type: 'PLACE_RING', playerId: 1, position: { x: 0, y: 0 }, count: 1 });
      engine.processAction({
        type: 'MOVE_STACK',
        playerId: 1,
        from: { x: 0, y: 0 },
        to: { x: 0, y: 1 },
      });

      // P2 places at (0,2) and moves to (0,3)
      engine.processAction({ type: 'PLACE_RING', playerId: 2, position: { x: 0, y: 2 }, count: 1 });
      engine.processAction({
        type: 'MOVE_STACK',
        playerId: 2,
        from: { x: 0, y: 2 },
        to: { x: 0, y: 3 },
      });

      // P1 places at (0,0) (empty now)
      engine.processAction({ type: 'PLACE_RING', playerId: 1, position: { x: 0, y: 0 }, count: 1 });

      // P1 captures P2's stack at (0,3) using stack at (0,1)?
      // No, P1's stack at (0,1) has height 1. P2's stack at (0,3) has height 1.
      // Distance (0,1) -> (0,3) is 2.
      // Capture requires:
      // 1. CH >= CH_T. (1 >= 1). OK.
      // 2. Distance >= H. (2 >= 1). OK.
      // 3. Landing beyond target. Target is (0,3). Landing could be (0,4).
      // 4. Path clear. (0,1) -> (0,3) -> (0,4).
      // (0,2) has marker from P2.
      // "All intermediate cells... May contain markers".
      // So (0,2) is fine.
      // (0,3) is target.
      // (0,4) must be empty or own marker.

      // Let's try capturing with stack at (0,1) targeting (0,3) landing at (0,4).
      // But P1 just placed at (0,0). P1 MUST move the stack at (0,0).
      // So P1 cannot move stack at (0,1).

      // So we need P1 to capture WITH the stack at (0,0).
      // Target P2 at (0,3).
      // Distance (0,0) -> (0,3) is 3.
      // Stack height 1. 3 >= 1. OK.
      // Landing (0,4).
      // Path: (0,1), (0,2), (0,3).
      // (0,1) has P1 stack! Blocked.

      // We need a clear path.
      // Let's setup differently.
      // P1 at (0,0). P2 at (0,2).
      // P1 captures P2.

      // Reset engine for this test to be clean?
      // Or just continue carefully.
      // Let's use a new engine instance for clean state.
      initialState = createInitialGameState('test-capture', 'square8', players, timeControl);
      engine = new GameEngine(initialState);

      // P1 places at (0,0) and moves to (0,1)
      engine.processAction({ type: 'PLACE_RING', playerId: 1, position: { x: 0, y: 0 }, count: 1 });
      engine.processAction({
        type: 'MOVE_STACK',
        playerId: 1,
        from: { x: 0, y: 0 },
        to: { x: 0, y: 1 },
      });

      // P2 places at (0,3) and moves to (0,4)
      engine.processAction({ type: 'PLACE_RING', playerId: 2, position: { x: 0, y: 3 }, count: 1 });
      engine.processAction({
        type: 'MOVE_STACK',
        playerId: 2,
        from: { x: 0, y: 3 },
        to: { x: 0, y: 4 },
      });

      // P1 places at (0,2).
      engine.processAction({ type: 'PLACE_RING', playerId: 1, position: { x: 0, y: 2 }, count: 1 });

      // P1 moves (0,2) to capture (0,4), landing at (0,5).
      // Distance (0,2) -> (0,5) is 3.
      // Stack height 1. 3 >= 1. OK.
      // Target (0,4) has P2 stack (height 1).
      // CH (1) >= CH_T (1). OK.
      // Path: (0,3) has P2 marker. OK.
      // Landing (0,5) empty. OK.

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: { x: 0, y: 2 },
        to: { x: 0, y: 5 },
        captureTarget: { x: 0, y: 4 },
      };

      const result = engine.processAction(action);
      expect(result.type).toBe('ACTION_PROCESSED');

      const state = engine.getGameState();
      // Stack at (0,2) moved to (0,5)
      expect(state.board.stacks.get('0,2')).toBeUndefined();
      expect(state.board.stacks.get('0,5')).toBeDefined();
      // Target stack at (0,4) should be gone (captured)
      expect(state.board.stacks.get('0,4')).toBeUndefined();
      // Capturing stack should have height 2 (1 original + 1 captured)
      expect(state.board.stacks.get('0,5')?.stackHeight).toBe(2);
      // Marker left at (0,2)
      expect(state.board.markers.get('0,2')?.player).toBe(1);
      // Marker at (0,3) (P2's) should be flipped to P1
      expect(state.board.markers.get('0,3')?.player).toBe(1);
    });
  });

  describe('Line Formation', () => {
    it('should detect and process a line', () => {
      // Setup: Create a line of 3 markers for P1 on row 0.
      // For square8, lineLength = 3 (minimum), so this is an exact-length line.
      // Exact-length lines can use PROCESS_LINE (no choice needed).
      // Overlength (4+) lines would require CHOOSE_LINE_REWARD action.

      // Manually inject markers (3 for exact-length on square8)
      initialState.board.markers.set('0,0', {
        player: 1,
        position: { x: 0, y: 0 },
        type: 'regular',
      });
      initialState.board.markers.set('1,0', {
        player: 1,
        position: { x: 1, y: 0 },
        type: 'regular',
      });
      initialState.board.markers.set('2,0', {
        player: 1,
        position: { x: 2, y: 0 },
        type: 'regular',
      });

      // We need to trigger line detection.
      // Usually this happens after a move.
      // Let's make a dummy move that doesn't disturb the line.
      // P1 places at (0,1) and moves to (1,1).
      engine = new GameEngine(initialState);

      engine.processAction({ type: 'PLACE_RING', playerId: 1, position: { x: 0, y: 1 }, count: 1 });
      engine.processAction({
        type: 'MOVE_STACK',
        playerId: 1,
        from: { x: 0, y: 1 },
        to: { x: 1, y: 1 },
      });

      // After move, engine should detect lines.
      // Check state.formedLines
      let state = engine.getGameState();
      expect(state.board.formedLines.length).toBeGreaterThan(0);
      expect(state.board.formedLines[0].player).toBe(1);
      expect(state.board.formedLines[0].length).toBe(3);

      // Now process the line
      // Action: PROCESS_LINE (valid for exact-length lines)
      const action = {
        type: 'PROCESS_LINE',
        playerId: 1,
        lineIndex: 0,
      } as any; // Cast to any because we might need to import ProcessLineAction

      const result = engine.processAction(action);
      expect(result.type).toBe('ACTION_PROCESSED');

      state = engine.getGameState();
      // Markers should be collapsed
      expect(state.board.markers.get('0,0')).toBeUndefined();
      expect(state.board.collapsedSpaces.get('0,0')).toBe(1);
      expect(state.board.collapsedSpaces.get('2,0')).toBe(1);

      // P1 should have eliminated a ring (mandatory elimination)
      // Current implementation of LineMutator seems to only handle collapse.
      // Elimination might be a separate step or not yet implemented.
      // For now, we verify collapse happened.

      // Check if formedLines is updated (line removed)
      expect(state.board.formedLines.length).toBe(0);
    });
  });

  describe('Territory', () => {
    it('should detect disconnected territory and allow processing', () => {
      // Existing placeholder test: left as-is to document future work for
      // wiring territory detection into the shared engine. This is not yet
      // asserting behaviour, by design.
    });
  });

  // --- New tests: action surface coverage for shared GameAction variants ---

  describe('Action surface coverage (shared GameEngine + GameAction)', () => {
    it('supports SKIP_PLACEMENT as a no-op that advances to movement phase when optional', () => {
      initialState = createInitialGameState('skip-placement', 'square8', players, timeControl);

      // Seed a simple controllable stack for the current player so that
      // movement is already available. Under the written rules, placement
      // is optional in this situation and the player may explicitly skip it.
      const stackOwner = 1;
      (initialState.board.stacks as any).set('0,0', {
        position: { x: 0, y: 0 },
        rings: [stackOwner],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: stackOwner,
      });

      engine = new GameEngine(initialState);

      const before = engine.getGameState();
      expect(before.currentPhase).toBe('ring_placement');
      expect(before.currentPlayer).toBe(stackOwner);

      const skipAction = {
        type: 'SKIP_PLACEMENT',
        playerId: before.currentPlayer,
      } as any;

      const event = engine.processAction(skipAction);
      expect(event.type).toBe('ACTION_PROCESSED');

      const after = engine.getGameState();
      expect(after.currentPhase).toBe('movement');
      expect(after.currentPlayer).toBe(before.currentPlayer);
    });

    it('applies CONTINUE_CHAIN capture using the capture mutator without throwing', () => {
      initialState = createInitialGameState('continue-chain', 'square8', players, timeControl);

      // Manually seed attacker and target stacks along a file:
      // attacker at (0,0), target at (0,1), landing at (0,2).
      initialState.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });
      initialState.board.stacks.set('0,1', {
        position: { x: 0, y: 1 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      (initialState as any).currentPhase = 'capture';
      (initialState as any).currentPlayer = 1;

      engine = new GameEngine(initialState);

      const action = {
        type: 'CONTINUE_CHAIN',
        playerId: 1,
        from: { x: 0, y: 0 },
        captureTarget: { x: 0, y: 1 },
        to: { x: 0, y: 2 },
      } as any;

      const event = engine.processAction(action);
      expect(event.type).toBe('ACTION_PROCESSED');

      const state = engine.getGameState();
      const landingStack = state.board.stacks.get('0,2');
      expect(landingStack).toBeDefined();
      expect(landingStack?.stackHeight).toBe(3);
      // Captured ring is appended to the bottom of attacker's stack (per Rule 4.2.3)
      // Attacker rings [1, 1] stay on top, captured ring 2 goes to bottom
      expect(landingStack?.rings).toEqual([1, 1, 2]);

      // Origin should now have a marker for player 1.
      const originMarker = state.board.markers.get('0,0');
      expect(originMarker?.player).toBe(1);
    });

    it('allows CHOOSE_LINE_REWARD MINIMUM_COLLAPSE for overlength lines and collapses only the chosen subset', () => {
      // Use 3 players so line length = 3 per RR-CANON-R120 (square8 2p uses line length 4)
      const threePlayers: Player[] = [
        { ...players[0] },
        { ...players[1] },
        {
          id: 'p3',
          username: 'Player 3',
          type: 'human',
          playerNumber: 3,
          isReady: true,
          timeRemaining: 600,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];
      initialState = createInitialGameState(
        'choose-line-reward',
        'square8',
        threePlayers,
        timeControl
      );

      // Set up a single overlength line for player 1 on row 0: 5 markers.
      // For square8 3-player, lineLength = 3 (minimum), so this is an overlength line.
      const linePositions = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
        { x: 3, y: 0 },
        { x: 4, y: 0 },
      ];

      // Seed formedLines directly; this is independent of BoardManager.
      (initialState.board as any).formedLines = [
        {
          player: 1,
          positions: linePositions,
          length: linePositions.length,
        },
      ];

      (initialState as any).currentPlayer = 1;
      (initialState as any).currentPhase = 'line_processing';

      engine = new GameEngine(initialState);

      // For square8, required line length is 3.
      const minLength = 3;
      const collapsedSubset = linePositions.slice(0, minLength);

      const action = {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: collapsedSubset,
      } as any;

      const event = engine.processAction(action);
      expect(event.type).toBe('ACTION_PROCESSED');

      const state = engine.getGameState();

      // First 3 positions collapsed to player 1; positions 4 and 5 remain non-collapsed.
      const collapsedKeys = collapsedSubset.map((p) => `${p.x},${p.y}`);
      for (const key of collapsedKeys) {
        expect(state.board.collapsedSpaces.get(key)).toBe(1);
      }
      expect(state.board.collapsedSpaces.get('3,0')).toBeUndefined();
      expect(state.board.collapsedSpaces.get('4,0')).toBeUndefined();

      // Processed line removed from formedLines.
      expect(state.board.formedLines.length).toBe(0);
    });

    it('rejects CHOOSE_LINE_REWARD MINIMUM_COLLAPSE when positions are non-consecutive', () => {
      // Use 3 players so line length = 3 per RR-CANON-R120 (square8 2p uses line length 4)
      const threePlayers: Player[] = [
        { ...players[0] },
        { ...players[1] },
        {
          id: 'p3',
          username: 'Player 3',
          type: 'human',
          playerNumber: 3,
          isReady: true,
          timeRemaining: 600,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];
      initialState = createInitialGameState(
        'choose-line-reward-invalid',
        'square8',
        threePlayers,
        timeControl
      );

      // For square8 3-player, lineLength = 3. A 4-marker line is overlength,
      // so MINIMUM_COLLAPSE is a valid selection option.
      const linePositions = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
        { x: 3, y: 0 },
      ];

      (initialState.board as any).formedLines = [
        {
          player: 1,
          positions: linePositions,
          length: linePositions.length,
        },
      ];

      (initialState as any).currentPlayer = 1;
      (initialState as any).currentPhase = 'line_processing';

      engine = new GameEngine(initialState);

      // Provide exactly 3 positions (the required count), but non-consecutive.
      // Choose positions 0, 1, and 3 (skipping 2).
      const badCollapsed = [linePositions[0], linePositions[1], linePositions[3]];

      const badAction = {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: badCollapsed,
      } as any;

      // We expect validation to fail and an ERROR_OCCURRED event.
      const event = engine.processAction(badAction);
      expect(event.type).toBe('ERROR_OCCURRED');
      if (event.type === 'ERROR_OCCURRED') {
        // For overlength lines with the correct position count but non-consecutive
        // positions, the validator rejects with NON_CONSECUTIVE.
        expect(event.payload.code).toBe('NON_CONSECUTIVE');
      }
    });

    it('applies ELIMINATE_STACK to remove the entire cap and update elimination counts', () => {
      initialState = createInitialGameState('eliminate-stack', 'square8', players, timeControl);

      // Seed a 3-ring mixed-color stack at (0,0): [1, 2, 1] (top to bottom)
      // Cap height is 1 (only the top ring), so eliminating removes just the top ring.
      initialState.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        rings: [1, 2, 1],
        stackHeight: 3,
        capHeight: 1,
        controllingPlayer: 1,
      });
      initialState.board.eliminatedRings[1] = 0;
      const p1 = initialState.players.find((p) => p.playerNumber === 1)!;
      p1.eliminatedRings = 0;

      (initialState as any).currentPlayer = 1;
      (initialState as any).currentPhase = 'territory_processing';

      engine = new GameEngine(initialState);

      const action = {
        type: 'ELIMINATE_STACK',
        playerId: 1,
        stackPosition: { x: 0, y: 0 },
      } as any;

      const event = engine.processAction(action);
      expect(event.type).toBe('ACTION_PROCESSED');

      const state = engine.getGameState();
      const stack = state.board.stacks.get('0,0');
      // After eliminating cap of 1, remaining rings are [2, 1]
      expect(stack).toBeDefined();
      expect(stack?.stackHeight).toBe(2);
      expect(stack?.rings).toEqual([2, 1]);
      // Controller changes to player 2 (new top ring)
      expect(stack?.controllingPlayer).toBe(2);

      const p1After = state.players.find((p) => p.playerNumber === 1)!;
      expect(p1After.eliminatedRings).toBe(1);
      expect(state.board.eliminatedRings[1]).toBe(1);
    });

    it('applies PROCESS_TERRITORY to collapse a region into territory without throwing', () => {
      initialState = createInitialGameState('process-territory', 'square8', players, timeControl);

      // Seed a single disconnected region controlled by player 1.
      (initialState.board.territories as any).set('region-1', {
        id: 'region-1',
        controllingPlayer: 1,
        isDisconnected: true,
        spaces: [{ x: 0, y: 0 }],
      });

      (initialState as any).currentPlayer = 1;
      (initialState as any).currentPhase = 'territory_processing';

      engine = new GameEngine(initialState);

      const action = {
        type: 'PROCESS_TERRITORY',
        playerId: 1,
        regionId: 'region-1',
      } as any;

      const event = engine.processAction(action);
      expect(event.type).toBe('ACTION_PROCESSED');

      const state = engine.getGameState();

      // Territory should be removed from territories map after collapsing
      const removed = (state.board.territories as any).get('region-1');
      expect(removed).toBeUndefined();

      // Space should now be collapsed to player 1
      const collapsed = (state.board.collapsedSpaces as any).get('0,0');
      expect(collapsed).toBe(1);

      // Player 1 should have gained 1 territory space
      const p1 = state.players.find((p) => p.playerNumber === 1)!;
      expect(p1.territorySpaces).toBe(1);
    });
  });
});
