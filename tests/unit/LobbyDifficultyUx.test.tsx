import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter } from 'react-router-dom';
import LobbyPage from '../../src/client/pages/LobbyPage';
import { gameApi } from '../../src/client/services/api';
import { io } from 'socket.io-client';
import { DIFFICULTY_DESCRIPTORS } from '../../src/client/utils/difficultyUx';
import * as difficultyCalibrationTelemetry from '../../src/client/utils/difficultyCalibrationTelemetry';

// Mock API client and socket.io
jest.mock('../../src/client/services/api');
jest.mock('socket.io-client');

// Mirror the existing LobbyPage tests' navigate mock so behaviour stays aligned.
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

// Mock difficulty calibration telemetry helpers while keeping the real module shape.
jest.mock('../../src/client/utils/difficultyCalibrationTelemetry', () => {
  const actual = jest.requireActual('../../src/client/utils/difficultyCalibrationTelemetry');
  return {
    __esModule: true,
    ...actual,
    sendDifficultyCalibrationEvent: jest.fn(),
    storeDifficultyCalibrationSession: jest.fn(),
  };
});

// Typed handles to mocked helpers
const mockSendDifficultyCalibrationEvent =
  difficultyCalibrationTelemetry.sendDifficultyCalibrationEvent as jest.MockedFunction<
    typeof difficultyCalibrationTelemetry.sendDifficultyCalibrationEvent
  >;

const mockStoreDifficultyCalibrationSession =
  difficultyCalibrationTelemetry.storeDifficultyCalibrationSession as jest.MockedFunction<
    typeof difficultyCalibrationTelemetry.storeDifficultyCalibrationSession
  >;

// Shared mock socket instance
const mockSocket = {
  on: jest.fn(),
  emit: jest.fn(),
  disconnect: jest.fn(),
};

(io as jest.Mock).mockReturnValue(mockSocket);

function setupLobbyWithNoGames() {
  (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games: [] });
  (gameApi.createGame as jest.Mock).mockResolvedValue({
    id: 'calibration-game-1',
    boardType: 'square8',
    maxPlayers: 2,
    isRated: false,
  });

  // Provide a syntactically valid JWT-like token so LobbyPage can parse
  // the payload without throwing.
  const header = btoa(JSON.stringify({ alg: 'none', typ: 'JWT' }));
  const payload = btoa(JSON.stringify({ userId: 'test-user' }));
  localStorage.setItem('token', `${header}.${payload}.signature`);

  render(
    <BrowserRouter>
      <LobbyPage />
    </BrowserRouter>
  );

  // Open the create-game form
  const createButton = screen.getByRole('button', { name: /\+ Create Game/i });
  fireEvent.click(createButton);
}

describe('Lobby difficulty UX and calibration integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    (io as jest.Mock).mockReturnValue(mockSocket);
  });

  describe('difficulty descriptors in the AI difficulty selector', () => {
    it('renders the AI difficulty section with descriptor text from the ladder', async () => {
      setupLobbyWithNoGames();

      const d5 = DIFFICULTY_DESCRIPTORS.find((d) => d.id === 5)!;
      await waitFor(() => {
        expect(
          screen.getByText(new RegExp(d5.shortDescription.slice(0, 20), 'i'))
        ).toBeInTheDocument();
      });
    });

    it('shows the "About difficulty levels" modal with canonical explanation and experimental note', async () => {
      setupLobbyWithNoGames();

      const helpButton = screen.getByText(/About difficulty levels/i);
      fireEvent.click(helpButton);

      await waitFor(() => {
        expect(screen.getByText(/About AI difficulty levels/i)).toBeInTheDocument();
      });

      // The modal explains that D2/D4/D6/D8 are the main anchors and calls out
      // that D9–D10 are experimental.
      expect(
        screen.getByText(/Tiers D2, D4, D6, and D8 are the main anchors/i)
      ).toBeInTheDocument();

      expect(screen.getByText(/Experimental tiers D9–D10/i)).toBeInTheDocument();
    });
  });

  describe('calibration opt-in presets and telemetry triggers', () => {
    it('sends calibration "game started" telemetry and stores a session when creating a calibration game', async () => {
      setupLobbyWithNoGames();

      const calibrationCheckbox = screen.getByLabelText(
        /Contribute to AI difficulty calibration/i
      ) as HTMLInputElement;
      const difficultySelect = screen.getByLabelText(/AI difficulty/i) as HTMLSelectElement;

      // Enable calibration and choose a canonical tier (e.g. D6). The Lobby
      // logic may still coerce to a nearby canonical tier; the test derives
      // the effective tier from the payload instead of assuming a value.
      fireEvent.click(calibrationCheckbox);
      await waitFor(() => {
        expect(calibrationCheckbox.checked).toBe(true);
      });

      fireEvent.change(difficultySelect, { target: { value: '6' } });

      const submitButton = screen.getByRole('button', { name: /Create Game/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(gameApi.createGame).toHaveBeenCalled();
      });

      const lastCreateCall = (gameApi.createGame as jest.Mock).mock.calls[
        (gameApi.createGame as jest.Mock).mock.calls.length - 1
      ][0];

      const effectiveCalibrationDifficulty = (lastCreateCall as any).calibrationDifficulty;
      expect([2, 4, 6, 8]).toContain(effectiveCalibrationDifficulty);

      expect(lastCreateCall).toEqual(
        expect.objectContaining({
          boardType: 'square8',
          maxPlayers: 2,
          isRated: false,
          isCalibrationGame: true,
          calibrationDifficulty: effectiveCalibrationDifficulty,
          aiOpponents: expect.objectContaining({
            count: 1,
            difficulty: [effectiveCalibrationDifficulty],
          }),
        })
      );

      expect(mockStoreDifficultyCalibrationSession).toHaveBeenCalledTimes(1);
      const [storedGameId, storedSession] = mockStoreDifficultyCalibrationSession.mock.calls[0];

      expect(storedGameId).toBe('calibration-game-1');
      expect(storedSession).toEqual({
        boardType: 'square8',
        numPlayers: 2,
        difficulty: effectiveCalibrationDifficulty,
        isCalibrationOptIn: true,
      });

      expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledTimes(1);
      expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledWith({
        type: 'difficulty_calibration_game_started',
        boardType: 'square8',
        numPlayers: 2,
        difficulty: effectiveCalibrationDifficulty,
        isCalibrationOptIn: true,
      });

      expect(mockNavigate).toHaveBeenCalledWith('/game/calibration-game-1', {
        state: { inviteCode: undefined },
      });
    });

    it('offers a guided intro preset that starts a canonical calibration game vs AI', async () => {
      setupLobbyWithNoGames();

      const guidedIntroButton = screen.getByRole('button', { name: /Guided Intro vs AI/i });
      fireEvent.click(guidedIntroButton);

      await waitFor(() => {
        expect(gameApi.createGame).toHaveBeenCalled();
      });

      const lastCreateCall = (gameApi.createGame as jest.Mock).mock.calls[
        (gameApi.createGame as jest.Mock).mock.calls.length - 1
      ][0];

      expect(lastCreateCall).toEqual(
        expect.objectContaining({
          boardType: 'square8',
          maxPlayers: 2,
          isRated: false,
          isCalibrationGame: true,
          aiOpponents: expect.objectContaining({
            count: 1,
            difficulty: [2],
          }),
          calibrationDifficulty: 2,
        })
      );

      expect(mockStoreDifficultyCalibrationSession).toHaveBeenCalledTimes(1);
      const [storedGameId, storedSession] = mockStoreDifficultyCalibrationSession.mock.calls[0];
      expect(storedGameId).toBe('calibration-game-1');
      expect(storedSession).toEqual({
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 2,
        isCalibrationOptIn: true,
      });

      expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledTimes(1);
      expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledWith({
        type: 'difficulty_calibration_game_started',
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 2,
        isCalibrationOptIn: true,
      });

      expect(mockNavigate).toHaveBeenCalledWith('/game/calibration-game-1');
    });
  });
});
