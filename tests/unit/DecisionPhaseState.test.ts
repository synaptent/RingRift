/**
 * Tests for DecisionPhaseState state machine
 * @module tests/unit/DecisionPhaseState.test
 */

import {
  createIdleState,
  initializeDecision,
  issueWarning,
  expireDecision,
  resolveDecision,
  cancelDecision,
  clearDecision,
  isDecisionActive,
  isDecisionTerminal,
  getRemainingTime,
  getDecisionMetadata,
  getTimeoutForChoice,
  getWarningScheduleTime,
  getExpiryScheduleTime,
  DEFAULT_TIMEOUT_CONFIG,
  type DecisionPhaseState,
  type DecisionPendingState,
  type InitializeDecisionParams,
} from '../../src/shared/decisions/DecisionPhaseState';

describe('DecisionPhaseState', () => {
  const baseParams: InitializeDecisionParams = {
    phase: 'line_processing',
    player: 1,
    choiceType: 'line_reward',
    choiceKind: 'line_reward_choice',
    timeoutMs: 30000,
    nowMs: 1000000,
  };

  describe('createIdleState', () => {
    it('should create an idle state', () => {
      const state = createIdleState();
      expect(state.kind).toBe('idle');
    });
  });

  describe('initializeDecision', () => {
    it('should create a pending state with correct values', () => {
      const state = initializeDecision(baseParams);

      expect(state.kind).toBe('pending');
      expect(state.phase).toBe('line_processing');
      expect(state.player).toBe(1);
      expect(state.choiceType).toBe('line_reward');
      expect(state.choiceKind).toBe('line_reward_choice');
      expect(state.startedAt).toBe(1000000);
      expect(state.deadlineMs).toBe(1030000);
      expect(state.timeoutMs).toBe(30000);
    });

    it('should use Date.now() when nowMs not provided', () => {
      const before = Date.now();
      const state = initializeDecision({
        ...baseParams,
        nowMs: undefined,
      });
      const after = Date.now();

      expect(state.startedAt).toBeGreaterThanOrEqual(before);
      expect(state.startedAt).toBeLessThanOrEqual(after);
    });

    it('should work with different choice types', () => {
      const chainState = initializeDecision({
        ...baseParams,
        choiceType: 'chain_capture',
        choiceKind: 'chain_capture_choice',
      });

      expect(chainState.choiceType).toBe('chain_capture');
      expect(chainState.choiceKind).toBe('chain_capture_choice');
    });
  });

  describe('issueWarning', () => {
    it('should transition from pending to warning state', () => {
      const pending = initializeDecision(baseParams);
      const warning = issueWarning(pending, 5000, 1025000);

      expect(warning.kind).toBe('warning_issued');
      expect(warning.phase).toBe(pending.phase);
      expect(warning.player).toBe(pending.player);
      expect(warning.choiceType).toBe(pending.choiceType);
      expect(warning.choiceKind).toBe(pending.choiceKind);
      expect(warning.startedAt).toBe(pending.startedAt);
      expect(warning.deadlineMs).toBe(pending.deadlineMs);
      expect(warning.remainingMsAtWarning).toBe(5000);
      expect(warning.warningIssuedAt).toBe(1025000);
    });

    it('should use Date.now() when nowMs not provided', () => {
      const pending = initializeDecision(baseParams);
      const before = Date.now();
      const warning = issueWarning(pending, 5000);
      const after = Date.now();

      expect(warning.warningIssuedAt).toBeGreaterThanOrEqual(before);
      expect(warning.warningIssuedAt).toBeLessThanOrEqual(after);
    });
  });

  describe('expireDecision', () => {
    it('should transition from pending to expired state', () => {
      const pending = initializeDecision(baseParams);
      const expired = expireDecision(pending, 1030000);

      expect(expired.kind).toBe('expired');
      expect(expired.phase).toBe(pending.phase);
      expect(expired.player).toBe(pending.player);
      expect(expired.choiceType).toBe(pending.choiceType);
      expect(expired.choiceKind).toBe(pending.choiceKind);
      expect(expired.expiredAt).toBe(1030000);
      expect(expired.deadlineMs).toBe(pending.deadlineMs);
    });

    it('should transition from warning to expired state', () => {
      const pending = initializeDecision(baseParams);
      const warning = issueWarning(pending, 5000, 1025000);
      const expired = expireDecision(warning, 1030000);

      expect(expired.kind).toBe('expired');
      expect(expired.phase).toBe(warning.phase);
      expect(expired.player).toBe(warning.player);
    });
  });

  describe('resolveDecision', () => {
    it('should resolve from pending state with player action', () => {
      const pending = initializeDecision(baseParams);
      const resolved = resolveDecision(pending, 'player_action', 'move-123', 1015000);

      expect(resolved.kind).toBe('resolved');
      expect(resolved.resolution).toBe('player_action');
      expect(resolved.phase).toBe(pending.phase);
      expect(resolved.player).toBe(pending.player);
      expect(resolved.choiceType).toBe(pending.choiceType);
      expect(resolved.resolvedMoveId).toBe('move-123');
      expect(resolved.resolvedAt).toBe(1015000);
      expect(resolved.durationMs).toBe(15000);
    });

    it('should resolve from warning state', () => {
      const pending = initializeDecision(baseParams);
      const warning = issueWarning(pending, 5000, 1025000);
      const resolved = resolveDecision(warning, 'player_action', 'move-456', 1028000);

      expect(resolved.kind).toBe('resolved');
      expect(resolved.resolution).toBe('player_action');
      expect(resolved.durationMs).toBe(28000);
    });

    it('should resolve from expired state with timeout resolution', () => {
      const pending = initializeDecision(baseParams);
      const expired = expireDecision(pending, 1030000);
      const resolved = resolveDecision(expired, 'timeout', undefined, 1030001);

      expect(resolved.kind).toBe('resolved');
      expect(resolved.resolution).toBe('timeout');
      expect(resolved.resolvedMoveId).toBeUndefined();
    });

    it('should resolve with auto_resolved', () => {
      const pending = initializeDecision(baseParams);
      const resolved = resolveDecision(pending, 'auto_resolved', undefined, 1010000);

      expect(resolved.resolution).toBe('auto_resolved');
    });

    it('should not include resolvedMoveId when undefined', () => {
      const pending = initializeDecision(baseParams);
      const resolved = resolveDecision(pending, 'player_action', undefined, 1015000);

      expect('resolvedMoveId' in resolved).toBe(false);
    });
  });

  describe('cancelDecision', () => {
    it('should cancel from pending state', () => {
      const pending = initializeDecision(baseParams);
      const cancelled = cancelDecision(pending, 'game_ended', 1020000);

      expect(cancelled.kind).toBe('cancelled');
      expect(cancelled.reason).toBe('game_ended');
      expect(cancelled.phase).toBe(pending.phase);
      expect(cancelled.player).toBe(pending.player);
      expect(cancelled.cancelledAt).toBe(1020000);
    });

    it('should cancel from idle state without phase/player', () => {
      const idle = createIdleState();
      const cancelled = cancelDecision(idle, 'no_decision_needed', 1000000);

      expect(cancelled.kind).toBe('cancelled');
      expect(cancelled.reason).toBe('no_decision_needed');
      expect(cancelled.phase).toBeUndefined();
      expect(cancelled.player).toBeUndefined();
    });

    it('should cancel from warning state', () => {
      const pending = initializeDecision(baseParams);
      const warning = issueWarning(pending, 5000, 1025000);
      const cancelled = cancelDecision(warning, 'player_disconnected', 1027000);

      expect(cancelled.kind).toBe('cancelled');
      expect(cancelled.phase).toBe(warning.phase);
      expect(cancelled.player).toBe(warning.player);
    });
  });

  describe('clearDecision', () => {
    it('should return idle state', () => {
      const state = clearDecision();
      expect(state.kind).toBe('idle');
    });
  });

  describe('isDecisionActive', () => {
    it('should return true for pending state', () => {
      const pending = initializeDecision(baseParams);
      expect(isDecisionActive(pending)).toBe(true);
    });

    it('should return true for warning state', () => {
      const pending = initializeDecision(baseParams);
      const warning = issueWarning(pending, 5000);
      expect(isDecisionActive(warning)).toBe(true);
    });

    it('should return false for idle state', () => {
      const idle = createIdleState();
      expect(isDecisionActive(idle)).toBe(false);
    });

    it('should return false for expired state', () => {
      const pending = initializeDecision(baseParams);
      const expired = expireDecision(pending, 1030000);
      expect(isDecisionActive(expired)).toBe(false);
    });

    it('should return false for resolved state', () => {
      const pending = initializeDecision(baseParams);
      const resolved = resolveDecision(pending, 'player_action');
      expect(isDecisionActive(resolved)).toBe(false);
    });

    it('should return false for cancelled state', () => {
      const pending = initializeDecision(baseParams);
      const cancelled = cancelDecision(pending, 'test');
      expect(isDecisionActive(cancelled)).toBe(false);
    });
  });

  describe('isDecisionTerminal', () => {
    it('should return true for expired state', () => {
      const pending = initializeDecision(baseParams);
      const expired = expireDecision(pending, 1030000);
      expect(isDecisionTerminal(expired)).toBe(true);
    });

    it('should return true for resolved state', () => {
      const pending = initializeDecision(baseParams);
      const resolved = resolveDecision(pending, 'player_action');
      expect(isDecisionTerminal(resolved)).toBe(true);
    });

    it('should return true for cancelled state', () => {
      const pending = initializeDecision(baseParams);
      const cancelled = cancelDecision(pending, 'test');
      expect(isDecisionTerminal(cancelled)).toBe(true);
    });

    it('should return false for idle state', () => {
      const idle = createIdleState();
      expect(isDecisionTerminal(idle)).toBe(false);
    });

    it('should return false for pending state', () => {
      const pending = initializeDecision(baseParams);
      expect(isDecisionTerminal(pending)).toBe(false);
    });

    it('should return false for warning state', () => {
      const pending = initializeDecision(baseParams);
      const warning = issueWarning(pending, 5000);
      expect(isDecisionTerminal(warning)).toBe(false);
    });
  });

  describe('getRemainingTime', () => {
    it('should return remaining time for pending state', () => {
      const pending = initializeDecision(baseParams);
      const remaining = getRemainingTime(pending, 1015000);

      expect(remaining).toBe(15000);
    });

    it('should return remaining time for warning state', () => {
      const pending = initializeDecision(baseParams);
      const warning = issueWarning(pending, 5000, 1025000);
      const remaining = getRemainingTime(warning, 1028000);

      expect(remaining).toBe(2000);
    });

    it('should return 0 when past deadline', () => {
      const pending = initializeDecision(baseParams);
      const remaining = getRemainingTime(pending, 1040000);

      expect(remaining).toBe(0);
    });

    it('should return null for idle state', () => {
      const idle = createIdleState();
      expect(getRemainingTime(idle)).toBeNull();
    });

    it('should return null for expired state', () => {
      const pending = initializeDecision(baseParams);
      const expired = expireDecision(pending, 1030000);
      expect(getRemainingTime(expired)).toBeNull();
    });

    it('should return null for resolved state', () => {
      const pending = initializeDecision(baseParams);
      const resolved = resolveDecision(pending, 'player_action');
      expect(getRemainingTime(resolved)).toBeNull();
    });
  });

  describe('getDecisionMetadata', () => {
    it('should return metadata for idle state', () => {
      const idle = createIdleState();
      const metadata = getDecisionMetadata(idle);

      expect(metadata).toEqual({ kind: 'idle' });
    });

    it('should return metadata for pending state', () => {
      const pending = initializeDecision(baseParams);
      const metadata = getDecisionMetadata(pending);

      expect(metadata.kind).toBe('pending');
      expect(metadata.phase).toBe('line_processing');
      expect(metadata.player).toBe(1);
      expect(metadata.choiceType).toBe('line_reward');
      expect(metadata.remainingMs).toBeDefined();
    });

    it('should return metadata for cancelled state with phase/player', () => {
      const pending = initializeDecision(baseParams);
      const cancelled = cancelDecision(pending, 'test');
      const metadata = getDecisionMetadata(cancelled);

      expect(metadata.kind).toBe('cancelled');
      expect(metadata.phase).toBe('line_processing');
      expect(metadata.player).toBe(1);
    });

    it('should return metadata for cancelled state without phase/player', () => {
      const idle = createIdleState();
      const cancelled = cancelDecision(idle, 'test');
      const metadata = getDecisionMetadata(cancelled);

      expect(metadata.kind).toBe('cancelled');
      expect(metadata.phase).toBeUndefined();
      expect(metadata.player).toBeUndefined();
    });

    it('should return metadata for resolved state', () => {
      const pending = initializeDecision(baseParams);
      const resolved = resolveDecision(pending, 'player_action');
      const metadata = getDecisionMetadata(resolved);

      expect(metadata.kind).toBe('resolved');
      expect(metadata.phase).toBe('line_processing');
      expect(metadata.player).toBe(1);
      expect(metadata.choiceType).toBe('line_reward');
    });
  });

  describe('getTimeoutForChoice', () => {
    it('should return base timeout for line_reward', () => {
      const timeout = getTimeoutForChoice('line_reward');
      expect(timeout).toBe(DEFAULT_TIMEOUT_CONFIG.baseTimeoutMs);
    });

    it('should return override timeout for chain_capture', () => {
      const timeout = getTimeoutForChoice('chain_capture');
      expect(timeout).toBe(15000);
    });

    it('should return override timeout for no_action', () => {
      const timeout = getTimeoutForChoice('no_action');
      expect(timeout).toBe(10000);
    });

    it('should use custom config', () => {
      const customConfig = {
        baseTimeoutMs: 60000,
        warningBeforeMs: 10000,
        choiceTypeOverrides: {
          line_reward: 45000,
        },
      };

      expect(getTimeoutForChoice('line_reward', customConfig)).toBe(45000);
      expect(getTimeoutForChoice('region_order', customConfig)).toBe(60000);
    });
  });

  describe('getWarningScheduleTime', () => {
    it('should calculate time until warning', () => {
      const pending = initializeDecision({
        ...baseParams,
        nowMs: Date.now(),
      });
      const scheduleTime = getWarningScheduleTime(pending);

      // Should be approximately timeoutMs - warningBeforeMs
      expect(scheduleTime).toBeGreaterThan(20000);
      expect(scheduleTime).toBeLessThanOrEqual(25000);
    });

    it('should return 0 if warning time has passed', () => {
      const pending = initializeDecision({
        ...baseParams,
        nowMs: Date.now() - 30000, // Started 30s ago
      });
      const scheduleTime = getWarningScheduleTime(pending);

      expect(scheduleTime).toBe(0);
    });
  });

  describe('getExpiryScheduleTime', () => {
    it('should calculate time until expiry', () => {
      const pending = initializeDecision({
        ...baseParams,
        nowMs: Date.now(),
      });
      const scheduleTime = getExpiryScheduleTime(pending);

      // Should be approximately timeoutMs
      expect(scheduleTime).toBeGreaterThan(29000);
      expect(scheduleTime).toBeLessThanOrEqual(30000);
    });

    it('should return 0 if expiry time has passed', () => {
      const pending = initializeDecision({
        ...baseParams,
        nowMs: Date.now() - 35000, // Started 35s ago
      });
      const scheduleTime = getExpiryScheduleTime(pending);

      expect(scheduleTime).toBe(0);
    });
  });

  describe('state machine flow', () => {
    it('should handle complete happy path: idle -> pending -> resolved', () => {
      let state: DecisionPhaseState = createIdleState();
      expect(state.kind).toBe('idle');

      state = initializeDecision(baseParams);
      expect(state.kind).toBe('pending');

      state = resolveDecision(state as DecisionPendingState, 'player_action', 'move-1', 1010000);
      expect(state.kind).toBe('resolved');
    });

    it('should handle warning path: pending -> warning -> resolved', () => {
      let state: DecisionPhaseState = initializeDecision(baseParams);
      expect(state.kind).toBe('pending');

      state = issueWarning(state as DecisionPendingState, 5000, 1025000);
      expect(state.kind).toBe('warning_issued');

      state = resolveDecision(state, 'player_action', 'move-2', 1028000);
      expect(state.kind).toBe('resolved');
    });

    it('should handle timeout path: pending -> warning -> expired -> resolved', () => {
      let state: DecisionPhaseState = initializeDecision(baseParams);

      state = issueWarning(state as DecisionPendingState, 5000, 1025000);
      state = expireDecision(state, 1030000);
      expect(state.kind).toBe('expired');

      state = resolveDecision(state, 'timeout', undefined, 1030001);
      expect(state.kind).toBe('resolved');
      expect((state as any).resolution).toBe('timeout');
    });

    it('should handle cancellation from any active state', () => {
      const pending = initializeDecision(baseParams);
      const cancelledFromPending = cancelDecision(pending, 'game_ended');
      expect(cancelledFromPending.kind).toBe('cancelled');

      const warning = issueWarning(pending, 5000);
      const cancelledFromWarning = cancelDecision(warning, 'player_left');
      expect(cancelledFromWarning.kind).toBe('cancelled');
    });
  });
});
