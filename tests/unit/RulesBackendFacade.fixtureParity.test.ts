import { GameState } from '../../src/shared/types/game';
import { computeProgressSnapshot, hashGameState } from '../../src/shared/engine/core';
import { RulesBackendFacade, RulesResult } from '../../src/server/game/RulesBackendFacade';
import * as envFlags from '../../src/shared/utils/envFlags';
import * as parity from '../../src/server/utils/rulesParityMetrics';
import { createTestGameState } from '../utils/fixtures';

describe('RulesBackendFacade – fixture-based parity via runPythonShadow', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    jest.spyOn(envFlags, 'getRulesMode').mockReturnValue('shadow');
  });

  it('does not increment parity metrics when Python echoes TS results', async () => {
    const tsBefore: GameState = createTestGameState();
    const tsAfter: GameState = { ...tsBefore };

    const tsResult: RulesResult = {
      success: true,
      gameState: tsAfter,
    };

    const tsHash = hashGameState(tsAfter as any);
    const tsProgress = computeProgressSnapshot(tsAfter as any);
    const tsS = tsProgress.S;
    const tsStatus = tsAfter.gameStatus;

    const pythonClient = {
      evaluateMove: jest.fn().mockResolvedValue({
        valid: tsResult.success,
        nextState: tsAfter,
        stateHash: tsHash,
        sInvariant: tsS,
        gameStatus: tsStatus,
      }),
    } as any;

    const engineStub = {} as any;
    const facade = new RulesBackendFacade(engineStub, pythonClient);

    const validIncSpy = jest.spyOn(parity.rulesParityMetrics.validMismatch, 'inc');
    const hashIncSpy = jest.spyOn(parity.rulesParityMetrics.hashMismatch, 'inc');
    const sIncSpy = jest.spyOn(parity.rulesParityMetrics.sMismatch, 'inc');
    const statusIncSpy = jest.spyOn(parity.rulesParityMetrics.gameStatusMismatch, 'inc');
    const logSpy = jest.spyOn(parity, 'logRulesMismatch').mockImplementation(() => {});

    const move = { type: 'move_stack', player: 1 } as any;

    await (facade as any).runPythonShadow(tsBefore, move, tsResult);

    expect(validIncSpy).not.toHaveBeenCalled();
    expect(hashIncSpy).not.toHaveBeenCalled();
    expect(sIncSpy).not.toHaveBeenCalled();
    expect(statusIncSpy).not.toHaveBeenCalled();
    expect(logSpy).not.toHaveBeenCalled();
  });

  it('increments the appropriate mismatch counters when Python diverges', async () => {
    const tsBefore: GameState = createTestGameState();
    const tsAfter: GameState = { ...tsBefore, gameStatus: 'completed' };

    const tsResult: RulesResult = {
      success: true,
      gameState: tsAfter,
    };

    const tsHash = hashGameState(tsAfter as any);
    const tsProgress = computeProgressSnapshot(tsAfter as any);
    const tsS = tsProgress.S;
    const tsStatus = tsAfter.gameStatus;

    const pythonClient = {
      evaluateMove: jest.fn().mockResolvedValue({
        valid: false,
        nextState: tsAfter,
        stateHash: `${tsHash}-py`, // hash mismatch
        sInvariant: tsS + 1, // S mismatch
        gameStatus: tsStatus === 'active' ? 'completed' : 'active', // status mismatch
      }),
    } as any;

    const engineStub = {} as any;
    const facade = new RulesBackendFacade(engineStub, pythonClient);

    const validIncSpy = jest.spyOn(parity.rulesParityMetrics.validMismatch, 'inc');
    const hashIncSpy = jest.spyOn(parity.rulesParityMetrics.hashMismatch, 'inc');
    const sIncSpy = jest.spyOn(parity.rulesParityMetrics.sMismatch, 'inc');
    const statusIncSpy = jest.spyOn(parity.rulesParityMetrics.gameStatusMismatch, 'inc');
    const logSpy = jest.spyOn(parity, 'logRulesMismatch').mockImplementation(() => {});

    const move = { type: 'move_stack', player: 1 } as any;

    await (facade as any).runPythonShadow(tsBefore, move, tsResult);

    expect(validIncSpy).toHaveBeenCalledTimes(1);
    expect(hashIncSpy).toHaveBeenCalledTimes(1);
    expect(sIncSpy).toHaveBeenCalledTimes(1);
    expect(statusIncSpy).toHaveBeenCalledTimes(1);

    expect(logSpy).toHaveBeenCalledWith(
      'valid',
      expect.objectContaining({
        tsValid: true,
        pyValid: false,
        mode: 'shadow',
      })
    );
    expect(logSpy).toHaveBeenCalledWith(
      'hash',
      expect.objectContaining({
        tsHash: tsHash,
        pyHash: `${tsHash}-py`,
        mode: 'shadow',
      })
    );
    expect(logSpy).toHaveBeenCalledWith(
      'S',
      expect.objectContaining({
        tsS,
        pyS: tsS + 1,
        mode: 'shadow',
      })
    );
    expect(logSpy).toHaveBeenCalledWith(
      'gameStatus',
      expect.objectContaining({
        tsStatus,
        pyStatus: tsStatus === 'active' ? 'completed' : 'active',
        mode: 'shadow',
      })
    );
  });
});
