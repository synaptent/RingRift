import { RulesBackendFacade } from '../../src/server/game/RulesBackendFacade';
import { computeProgressSnapshot, hashGameState } from '../../src/shared/engine/core';
import * as envFlags from '../../src/shared/utils/envFlags';
import * as parity from '../../src/server/utils/rulesParityMetrics';
import { createTestGameState } from '../utils/fixtures';

describe('RulesBackendFacade', () => {
  function makeFakeEngine() {
    return {
      getGameState: jest.fn(),
      makeMove: jest.fn(),
      makeMoveById: jest.fn(),
      getValidMoves: jest.fn(),
    } as any;
  }

  function makeFakePythonClient() {
    return {
      evaluateMove: jest.fn(),
    } as any;
  }

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('uses TS GameEngine only in ts mode', async () => {
    const engine = makeFakeEngine();
    const pythonClient = makeFakePythonClient();

    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('ts');
    jest.spyOn(envFlags, 'isRulesShadowMode').mockReturnValue(false);
    jest.spyOn(envFlags, 'isPythonRulesMode').mockReturnValue(false);
    const logSpy = jest.spyOn(parity, 'logRulesMismatch').mockImplementation(() => {});

    const tsResult = { success: true, gameState: { id: 'after' } as any };
    (engine.makeMove as jest.Mock).mockResolvedValue(tsResult);

    const facade = new RulesBackendFacade(engine as any, pythonClient);
    const move = { type: 'move_stack', player: 1 } as any;

    const result = await facade.applyMove(move);

    expect(engine.makeMove).toHaveBeenCalledTimes(1);
    expect(engine.makeMove).toHaveBeenCalledWith(move);
    expect(pythonClient.evaluateMove).not.toHaveBeenCalled();
    expect(logSpy).not.toHaveBeenCalled();
    expect(result).toBe(tsResult);
  });

  it('calls Python rules in shadow mode without blocking TS result and logs shadow_error on failure', async () => {
    const engine = makeFakeEngine();
    const pythonClient = makeFakePythonClient();

    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('shadow');
    jest.spyOn(envFlags, 'isRulesShadowMode').mockReturnValue(true);
    jest.spyOn(envFlags, 'isPythonRulesMode').mockReturnValue(false);
    const logSpy = jest.spyOn(parity, 'logRulesMismatch').mockImplementation(() => {});

    const beforeState = { id: 'before' } as any;
    (engine.getGameState as jest.Mock).mockReturnValue(beforeState);

    const tsResult = { success: true, gameState: { id: 'after' } as any };
    (engine.makeMove as jest.Mock).mockResolvedValue(tsResult);

    const error = new Error('py-fail');
    (pythonClient.evaluateMove as jest.Mock).mockRejectedValue(error);

    const facade = new RulesBackendFacade(engine as any, pythonClient);
    const move = { type: 'move_stack', player: 1 } as any;

    const result = await facade.applyMove(move);

    expect(result).toBe(tsResult);
    expect(engine.makeMove).toHaveBeenCalledTimes(1);
    expect(pythonClient.evaluateMove).toHaveBeenCalledTimes(1);
    expect(pythonClient.evaluateMove).toHaveBeenCalledWith(
      beforeState,
      expect.objectContaining(move)
    );

    // Allow the shadow promise chain to settle.
    await new Promise((resolve) => setImmediate(resolve));

    expect(logSpy).toHaveBeenCalledWith(
      'shadow_error',
      expect.objectContaining({
        error: expect.stringContaining('py-fail'),
      })
    );
  });

  it('uses Python as validation gate and TS GameEngine for state in python mode when move is valid', async () => {
    const engine = makeFakeEngine();
    const pythonClient = makeFakePythonClient();

    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('python');
    jest.spyOn(envFlags, 'isRulesShadowMode').mockReturnValue(false);
    const logSpy = jest.spyOn(parity, 'logRulesMismatch').mockImplementation(() => {});

    const beforeState = createTestGameState();
    (engine.getGameState as jest.Mock).mockReturnValue(beforeState);

    const tsAfter = createTestGameState();
    const tsResult = { success: true, gameState: tsAfter };
    (engine.makeMove as jest.Mock).mockResolvedValue(tsResult);

    const tsHash = hashGameState(tsAfter as any);
    const tsProgress = computeProgressSnapshot(tsAfter as any);
    const tsS = tsProgress.S;
    const tsStatus = tsAfter.gameStatus;

    (pythonClient.evaluateMove as jest.Mock).mockResolvedValue({
      valid: true,
      validationError: undefined,
      nextState: tsAfter,
      stateHash: tsHash,
      sInvariant: tsS,
      gameStatus: tsStatus,
    });

    const facade = new RulesBackendFacade(engine as any, pythonClient);

    const move = { type: 'move_stack', player: 1 } as any;

    const result = await facade.applyMove(move);

    expect(pythonClient.evaluateMove).toHaveBeenCalledTimes(1);
    expect(pythonClient.evaluateMove).toHaveBeenCalledWith(
      beforeState,
      expect.objectContaining({
        type: 'move_stack',
        player: 1,
      })
    );
    // When Python returns nextState, the facade currently invokes
    // engine.makeMove once in the provisional py.nextState branch (shadow
    // comparison) and once in the canonical apply path. Both calls see the
    // same move payload and the final result is the TS GameEngine state.
    expect(engine.makeMove).toHaveBeenCalledTimes(2);
    expect(engine.makeMove).toHaveBeenCalledWith(move);
    expect(result).toBe(tsResult);
    expect(logSpy).not.toHaveBeenCalled();
  });

  it('returns an invalid result without mutating TS state when Python rejects the move in python mode', async () => {
    const engine = makeFakeEngine();
    const pythonClient = makeFakePythonClient();

    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('python');
    jest.spyOn(envFlags, 'isRulesShadowMode').mockReturnValue(false);
    const logSpy = jest.spyOn(parity, 'logRulesMismatch').mockImplementation(() => {});

    const beforeState = createTestGameState();
    (engine.getGameState as jest.Mock).mockReturnValue(beforeState);

    (pythonClient.evaluateMove as jest.Mock).mockResolvedValue({
      valid: false,
      validationError: 'Move not found in legal_moves',
      nextState: undefined,
      stateHash: undefined,
      sInvariant: undefined,
      gameStatus: beforeState.gameStatus,
    });

    const facade = new RulesBackendFacade(engine as any, pythonClient);
    const move = { type: 'move_stack', player: 1 } as any;

    const result = await facade.applyMove(move);

    expect(result.success).toBe(false);
    expect(result.error).toContain('Move not found in legal_moves');
    expect(engine.makeMove).not.toHaveBeenCalled();
    expect(logSpy).not.toHaveBeenCalled();
  });

  it('falls back to TS GameEngine and logs backend_fallback when Python evaluateMove throws in python mode', async () => {
    const engine = makeFakeEngine();
    const pythonClient = makeFakePythonClient();

    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('python');
    jest.spyOn(envFlags, 'isRulesShadowMode').mockReturnValue(false);
    const logSpy = jest.spyOn(parity, 'logRulesMismatch').mockImplementation(() => {});

    const beforeState = createTestGameState();
    (engine.getGameState as jest.Mock).mockReturnValue(beforeState);

    const tsResult = { success: true, gameState: createTestGameState() };
    (engine.makeMove as jest.Mock).mockResolvedValue(tsResult);

    const error = new Error('py-fail');
    (pythonClient.evaluateMove as jest.Mock).mockRejectedValue(error);

    const facade = new RulesBackendFacade(engine as any, pythonClient);
    const move = { type: 'move_stack', player: 1 } as any;

    const result = await facade.applyMove(move);

    expect(result).toBe(tsResult);
    expect(engine.makeMove).toHaveBeenCalledTimes(1);
    expect(pythonClient.evaluateMove).toHaveBeenCalledTimes(1);

    expect(logSpy).toHaveBeenCalledWith(
      'backend_fallback',
      expect.objectContaining({
        note: expect.stringContaining('RINGRIFT_RULES_MODE=python'),
        error: expect.stringContaining('py-fail'),
      })
    );
  });

  it('applyMoveById uses TS engine and calls Python in shadow mode when move succeeds', async () => {
    const engine = makeFakeEngine();
    const pythonClient = makeFakePythonClient();

    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('shadow');
    jest.spyOn(envFlags, 'isRulesShadowMode').mockReturnValue(true);
    jest.spyOn(envFlags, 'isPythonRulesMode').mockReturnValue(false);
    const logSpy = jest.spyOn(parity, 'logRulesMismatch').mockImplementation(() => {});

    const beforeState = { id: 'before' } as any;
    (engine.getGameState as jest.Mock).mockReturnValue(beforeState);

    const lastMove = { id: 'm1', type: 'move_stack', player: 1 } as any;
    const afterState = { id: 'after', moveHistory: [lastMove] } as any;
    const tsResult = { success: true, gameState: afterState };
    (engine.makeMoveById as jest.Mock).mockResolvedValue(tsResult);
    // Must mock getValidMoves so applyMoveById can resolve the move by id
    (engine.getValidMoves as jest.Mock).mockReturnValue([lastMove]);

    (pythonClient.evaluateMove as jest.Mock).mockRejectedValue(new Error('py-fail'));

    const facade = new RulesBackendFacade(engine as any, pythonClient);

    const result = await facade.applyMoveById(1, 'm1');

    expect(result).toBe(tsResult);
    expect(engine.makeMoveById).toHaveBeenCalledTimes(1);
    expect(engine.makeMoveById).toHaveBeenCalledWith(1, 'm1');

    await new Promise((resolve) => setImmediate(resolve));

    expect(pythonClient.evaluateMove).toHaveBeenCalledTimes(1);
    expect(pythonClient.evaluateMove).toHaveBeenCalledWith(beforeState, lastMove);
    expect(logSpy).toHaveBeenCalledWith(
      'shadow_error',
      expect.objectContaining({
        error: expect.any(String),
      })
    );
  });

  it('applyMoveById uses Python as validation gate in python mode before applying TS move', async () => {
    const engine = makeFakeEngine();
    const pythonClient = makeFakePythonClient();

    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('python');
    jest.spyOn(envFlags, 'isRulesShadowMode').mockReturnValue(false);
    const logSpy = jest.spyOn(parity, 'logRulesMismatch').mockImplementation(() => {});

    const beforeState = createTestGameState();
    (engine.getGameState as jest.Mock).mockReturnValue(beforeState);

    const candidateMove = {
      id: 'm1',
      type: 'process_line',
      player: 1,
      timestamp: new Date(),
      moveNumber: 1,
      thinkTime: 0,
    } as any;
    (engine.getValidMoves as jest.Mock).mockReturnValue([candidateMove]);

    const tsAfter = createTestGameState();
    const tsResult = { success: true, gameState: tsAfter };
    (engine.makeMoveById as jest.Mock).mockResolvedValue(tsResult);

    const tsHash = hashGameState(tsAfter as any);
    const tsProgress = computeProgressSnapshot(tsAfter as any);
    const tsS = tsProgress.S;
    const tsStatus = tsAfter.gameStatus;

    (pythonClient.evaluateMove as jest.Mock).mockResolvedValue({
      valid: true,
      validationError: undefined,
      nextState: tsAfter,
      stateHash: tsHash,
      sInvariant: tsS,
      gameStatus: tsStatus,
    });

    const facade = new RulesBackendFacade(engine as any, pythonClient);

    const result = await facade.applyMoveById(1, 'm1');

    expect(pythonClient.evaluateMove).toHaveBeenCalledTimes(1);
    expect(pythonClient.evaluateMove).toHaveBeenCalledWith(beforeState, candidateMove);
    expect(engine.makeMoveById).toHaveBeenCalledTimes(1);
    expect(engine.makeMoveById).toHaveBeenCalledWith(1, 'm1');
    expect(result).toBe(tsResult);
    expect(logSpy).not.toHaveBeenCalled();
  });

  it('records rules parity mismatches under the runtime_python_mode suite in python mode', async () => {
    const engine = makeFakeEngine();
    const pythonClient = makeFakePythonClient();

    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('python');
    jest.spyOn(envFlags, 'isRulesShadowMode').mockReturnValue(false);
    const metricsSpy = jest
      .spyOn(parity, 'recordRulesParityMismatch')
      .mockImplementation(() => {});

    const beforeState = createTestGameState();
    (engine.getGameState as jest.Mock).mockReturnValue(beforeState);

    const tsAfter = createTestGameState();
    const tsResult = { success: true, gameState: tsAfter };
    (engine.makeMove as jest.Mock).mockResolvedValue(tsResult);

    const tsHash = hashGameState(tsAfter as any);
    const tsProgress = computeProgressSnapshot(tsAfter as any);
    const tsS = tsProgress.S;
    const tsStatus = tsAfter.gameStatus;

    // Intentionally introduce a hash mismatch while keeping other fields aligned.
    (pythonClient.evaluateMove as jest.Mock).mockResolvedValue({
      valid: true,
      validationError: undefined,
      nextState: tsAfter,
      stateHash: `${tsHash}-mismatch`,
      sInvariant: tsS,
      gameStatus: tsStatus,
    });

    const facade = new RulesBackendFacade(engine as any, pythonClient as any);
    const move = { type: 'move_stack', player: 1 } as any;

    const result = await facade.applyMove(move);

    expect(result).toBe(tsResult);
    expect(engine.makeMove).toHaveBeenCalled();
    expect(pythonClient.evaluateMove).toHaveBeenCalled();

    expect(metricsSpy).toHaveBeenCalledWith({
      mismatchType: 'hash',
      suite: 'runtime_python_mode',
    });
  });

  it('records rules parity mismatches under the runtime_shadow suite in shadow mode', async () => {
    const engine = makeFakeEngine();
    const pythonClient = makeFakePythonClient();

    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('shadow');
    jest.spyOn(envFlags, 'isRulesShadowMode').mockReturnValue(true);
    const metricsSpy = jest
      .spyOn(parity, 'recordRulesParityMismatch')
      .mockImplementation(() => {});

    const beforeState = createTestGameState();
    (engine.getGameState as jest.Mock).mockReturnValue(beforeState);

    const tsAfter = createTestGameState();
    const tsResult = { success: true, gameState: tsAfter };
    (engine.makeMove as jest.Mock).mockResolvedValue(tsResult);

    const tsHash = hashGameState(tsAfter as any);
    const tsProgress = computeProgressSnapshot(tsAfter as any);
    const tsS = tsProgress.S;
    const tsStatus = tsAfter.gameStatus;

    // Intentionally introduce a hash mismatch while keeping other fields aligned.
    (pythonClient.evaluateMove as jest.Mock).mockResolvedValue({
      valid: true,
      validationError: undefined,
      nextState: tsAfter,
      stateHash: `${tsHash}-mismatch`,
      sInvariant: tsS,
      gameStatus: tsStatus,
    });

    const facade = new RulesBackendFacade(engine as any, pythonClient as any);
    const move = { type: 'move_stack', player: 1 } as any;

    const result = await facade.applyMove(move);

    expect(result).toBe(tsResult);
    expect(engine.makeMove).toHaveBeenCalledTimes(1);
    expect(pythonClient.evaluateMove).toHaveBeenCalledTimes(1);

    // Allow the shadow parity call to complete.
    await new Promise((resolve) => setImmediate(resolve));

    expect(metricsSpy).toHaveBeenCalledWith({
      mismatchType: 'hash',
      suite: 'runtime_shadow',
    });
  });
});
