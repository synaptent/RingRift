import React from 'react';
import { renderHook, act } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock the envFlags module before importing SandboxContext
jest.mock('../../../src/shared/utils/envFlags', () => ({
  flagEnabled: jest.fn(() => false),
  isTestEnvironment: jest.fn(() => true),
  isSandboxAiStallDiagnosticsEnabled: jest.fn(() => false),
}));

// Mock ClientSandboxEngine to prevent importing the entire engine
jest.mock('../../../src/client/sandbox/ClientSandboxEngine', () => ({
  ClientSandboxEngine: jest.fn().mockImplementation(() => ({
    getGameState: jest.fn(() => null),
    start: jest.fn(),
  })),
}));

import {
  SandboxProvider,
  useSandbox,
  type SandboxMode,
} from '../../../src/client/contexts/SandboxContext';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

describe('SandboxContext - Sandbox Mode Toggle', () => {
  beforeEach(() => {
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <SandboxProvider>{children}</SandboxProvider>
  );

  describe('Initial Mode', () => {
    it('defaults to beginner mode when localStorage is empty', () => {
      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.sandboxMode).toBe('beginner');
      expect(result.current.isBeginnerMode).toBe(true);
    });

    it('loads debug mode from localStorage when set', () => {
      localStorageMock.getItem.mockReturnValueOnce('debug');

      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.sandboxMode).toBe('debug');
      expect(result.current.isBeginnerMode).toBe(false);
    });

    it('defaults to beginner mode for invalid localStorage value', () => {
      localStorageMock.getItem.mockReturnValueOnce('invalid_mode');

      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.sandboxMode).toBe('beginner');
      expect(result.current.isBeginnerMode).toBe(true);
    });
  });

  describe('Mode Switching', () => {
    it('switches from beginner to debug mode', () => {
      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.sandboxMode).toBe('beginner');

      act(() => {
        result.current.setSandboxMode('debug');
      });

      expect(result.current.sandboxMode).toBe('debug');
      expect(result.current.isBeginnerMode).toBe(false);
    });

    it('switches from debug to beginner mode', () => {
      localStorageMock.getItem.mockReturnValueOnce('debug');

      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.sandboxMode).toBe('debug');

      act(() => {
        result.current.setSandboxMode('beginner');
      });

      expect(result.current.sandboxMode).toBe('beginner');
      expect(result.current.isBeginnerMode).toBe(true);
    });
  });

  describe('localStorage Persistence', () => {
    it('persists mode change to localStorage', () => {
      const { result } = renderHook(() => useSandbox(), { wrapper });

      act(() => {
        result.current.setSandboxMode('debug');
      });

      expect(localStorageMock.setItem).toHaveBeenCalledWith('ringrift_sandbox_mode', 'debug');
    });

    it('persists beginner mode to localStorage', () => {
      localStorageMock.getItem.mockReturnValueOnce('debug');

      const { result } = renderHook(() => useSandbox(), { wrapper });

      act(() => {
        result.current.setSandboxMode('beginner');
      });

      expect(localStorageMock.setItem).toHaveBeenCalledWith('ringrift_sandbox_mode', 'beginner');
    });
  });

  describe('isBeginnerMode Convenience Flag', () => {
    it('returns true when mode is beginner', () => {
      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.isBeginnerMode).toBe(true);
    });

    it('returns false when mode is debug', () => {
      const { result } = renderHook(() => useSandbox(), { wrapper });

      act(() => {
        result.current.setSandboxMode('debug');
      });

      expect(result.current.isBeginnerMode).toBe(false);
    });

    it('updates correctly when mode changes', () => {
      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.isBeginnerMode).toBe(true);

      act(() => {
        result.current.setSandboxMode('debug');
      });

      expect(result.current.isBeginnerMode).toBe(false);

      act(() => {
        result.current.setSandboxMode('beginner');
      });

      expect(result.current.isBeginnerMode).toBe(true);
    });
  });
});

describe('SandboxContext - Context Value Stability', () => {
  beforeEach(() => {
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <SandboxProvider>{children}</SandboxProvider>
  );

  it('provides sandboxMode and setSandboxMode in context', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current).toHaveProperty('sandboxMode');
    expect(result.current).toHaveProperty('setSandboxMode');
    expect(result.current).toHaveProperty('isBeginnerMode');
    expect(typeof result.current.setSandboxMode).toBe('function');
  });

  it('throws when useSandbox is called outside SandboxProvider', () => {
    // Suppress console.error for this test since React will log an error
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      renderHook(() => useSandbox());
    }).toThrow('useSandbox must be used within a SandboxProvider');

    consoleSpy.mockRestore();
  });
});

describe('SandboxContext - Developer Tools Toggle', () => {
  beforeEach(() => {
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <SandboxProvider>{children}</SandboxProvider>
  );

  it('provides developerToolsEnabled and its setter', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current).toHaveProperty('developerToolsEnabled');
    expect(result.current).toHaveProperty('setDeveloperToolsEnabled');
    expect(typeof result.current.setDeveloperToolsEnabled).toBe('function');
  });

  it('developerToolsEnabled defaults to false in beginner mode', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current.sandboxMode).toBe('beginner');
    expect(result.current.developerToolsEnabled).toBe(false);
  });

  it('can toggle developerToolsEnabled', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    act(() => {
      result.current.setDeveloperToolsEnabled(true);
    });

    expect(result.current.developerToolsEnabled).toBe(true);

    act(() => {
      result.current.setDeveloperToolsEnabled(false);
    });

    expect(result.current.developerToolsEnabled).toBe(false);
  });

  it('enables developer tools automatically in debug mode', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    act(() => {
      result.current.setSandboxMode('debug');
    });

    expect(result.current.sandboxMode).toBe('debug');
    expect(result.current.developerToolsEnabled).toBe(true);
  });

  it('disables developer tools automatically when switching back to beginner mode', () => {
    localStorageMock.getItem.mockReturnValue('debug');

    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current.sandboxMode).toBe('debug');
    expect(result.current.developerToolsEnabled).toBe(true);

    act(() => {
      result.current.setSandboxMode('beginner');
    });

    expect(result.current.sandboxMode).toBe('beginner');
    expect(result.current.developerToolsEnabled).toBe(false);
  });
});

describe('SandboxContext - Engine State Management', () => {
  beforeEach(() => {
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <SandboxProvider>{children}</SandboxProvider>
  );

  it('starts with no sandbox engine configured', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current.sandboxEngine).toBeNull();
    expect(result.current.isConfigured).toBe(false);
    expect(result.current.getSandboxGameState()).toBeNull();
  });

  it('provides resetSandboxEngine function', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(typeof result.current.resetSandboxEngine).toBe('function');
  });

  it('provides sandboxStateVersion for re-render tracking', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(typeof result.current.sandboxStateVersion).toBe('number');
    expect(result.current.sandboxStateVersion).toBe(0);
  });

  it('can increment sandboxStateVersion', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current.sandboxStateVersion).toBe(0);

    act(() => {
      result.current.setSandboxStateVersion((v) => v + 1);
    });

    expect(result.current.sandboxStateVersion).toBe(1);
  });
});

describe('SandboxContext - Stall Warning State', () => {
  beforeEach(() => {
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <SandboxProvider>{children}</SandboxProvider>
  );

  it('starts with no stall warning', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current.sandboxStallWarning).toBeNull();
  });

  it('can set and clear stall warning', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    act(() => {
      result.current.setSandboxStallWarning('AI stall detected');
    });

    expect(result.current.sandboxStallWarning).toBe('AI stall detected');

    act(() => {
      result.current.setSandboxStallWarning(null);
    });

    expect(result.current.sandboxStallWarning).toBeNull();
  });
});

describe('SandboxContext - Default Configuration', () => {
  beforeEach(() => {
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <SandboxProvider>{children}</SandboxProvider>
  );

  it('has default config with 2 players', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current.config.numPlayers).toBe(2);
  });

  it('has default config with square8 board', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current.config.boardType).toBe('square8');
  });

  it('has default player types of human, human, ai, ai', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current.config.playerTypes).toEqual(['human', 'human', 'ai', 'ai']);
  });

  it('has default AI difficulties of 4 for all slots', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    expect(result.current.config.aiDifficulties).toEqual([4, 4, 4, 4]);
  });

  it('can update config', () => {
    const { result } = renderHook(() => useSandbox(), { wrapper });

    act(() => {
      result.current.setConfig((prev) => ({
        ...prev,
        numPlayers: 3,
        boardType: 'square19',
      }));
    });

    expect(result.current.config.numPlayers).toBe(3);
    expect(result.current.config.boardType).toBe('square19');
  });
});
