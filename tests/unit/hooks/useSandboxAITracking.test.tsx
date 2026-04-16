import { render, screen, waitFor } from '@testing-library/react';
import { useSandboxAITracking } from '../../../src/client/hooks/useSandboxAITracking';
import type { ClientSandboxEngine } from '../../../src/client/sandbox/ClientSandboxEngine';
import type { GameState } from '../../../src/shared/types/game';
import { createTestGameState, createTestPlayer } from '../../utils/fixtures';

jest.mock('react-hot-toast', () => ({
  toast: {
    error: jest.fn(),
    success: jest.fn(),
  },
}));

function createAiTurnState(overrides: Partial<GameState> = {}): GameState {
  return createTestGameState({
    currentPlayer: 1,
    currentPhase: 'ring_placement',
    gameStatus: 'active',
    players: [
      createTestPlayer(1, { type: 'ai', username: 'AI 1' }),
      createTestPlayer(2, { type: 'ai', username: 'AI 2' }),
    ],
    ...overrides,
  });
}

function Harness({
  gameState,
  engine,
  onRunAi,
}: {
  gameState: GameState | null;
  engine: ClientSandboxEngine | null;
  onRunAi: (engineOverride?: ClientSandboxEngine) => void;
}) {
  const { state } = useSandboxAITracking(engine, gameState, onRunAi);

  return (
    <div data-testid="thinking-started-at">
      {state.aiThinkingStartedAt === null ? 'null' : String(state.aiThinkingStartedAt)}
    </div>
  );
}

describe('useSandboxAITracking', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    jest.useRealTimers();
  });

  it('resets thinking timer for each AI turn in AI-vs-AI games', async () => {
    let now = 1000;
    jest.spyOn(Date, 'now').mockImplementation(() => now);

    const engine = {} as ClientSandboxEngine;
    const onRunAi = jest.fn();
    const firstTurn = createAiTurnState();
    const { rerender } = render(
      <Harness gameState={firstTurn} engine={engine} onRunAi={onRunAi} />
    );

    await waitFor(() => {
      expect(screen.getByTestId('thinking-started-at')).toHaveTextContent('1000');
    });

    now = 2000;
    const secondTurn = createAiTurnState({
      moveHistory: [
        {
          id: 'move-1',
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ],
      history: [
        {
          id: 'move-1',
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ],
    });

    rerender(<Harness gameState={secondTurn} engine={engine} onRunAi={onRunAi} />);

    await waitFor(() => {
      expect(screen.getByTestId('thinking-started-at')).toHaveTextContent('2000');
    });
  });
});
