import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { ScenarioPickerModal } from '../../../src/client/components/ScenarioPickerModal';
import type { ScenarioPickerModalProps } from '../../../src/client/components/ScenarioPickerModal';
import type { LoadableScenario } from '../../../src/client/sandbox/scenarioTypes';
import * as scenarioLoader from '../../../src/client/sandbox/scenarioLoader';
import * as statePersistence from '../../../src/client/sandbox/statePersistence';

// Mock scenarioLoader module
jest.mock('../../../src/client/sandbox/scenarioLoader', () => ({
  loadVectorScenarios: jest.fn(),
  loadCuratedScenarios: jest.fn(),
  loadCustomScenarios: jest.fn(),
  deleteCustomScenario: jest.fn(),
  filterScenarios: jest.fn((scenarios) => scenarios),
}));

// Mock statePersistence module
jest.mock('../../../src/client/sandbox/statePersistence', () => ({
  importScenarioFromFile: jest.fn(),
  exportScenarioToFile: jest.fn(),
}));

const mockLoadVectorScenarios = scenarioLoader.loadVectorScenarios as jest.Mock;
const mockLoadCuratedScenarios = scenarioLoader.loadCuratedScenarios as jest.Mock;
const mockLoadCustomScenarios = scenarioLoader.loadCustomScenarios as jest.Mock;
const mockDeleteCustomScenario = scenarioLoader.deleteCustomScenario as jest.Mock;
const mockFilterScenarios = scenarioLoader.filterScenarios as jest.Mock;
const mockExportScenarioToFile = statePersistence.exportScenarioToFile as jest.Mock;

const createMockScenario = (
  id: string,
  name: string,
  category: string = 'basics'
): LoadableScenario => ({
  id,
  name,
  description: `Description for ${name}`,
  category: category as any,
  boardType: 'square8',
  playerCount: 2,
  tags: [],
  difficulty: 'beginner',
  // Minimal metadata required by ScenarioMetadata/LoadableScenario
  createdAt: '2025-01-01T00:00:00.000Z',
  source: 'custom',
  // Serialized game state payload (structure not relevant for these tests)
  state: {} as any,
});

let consoleErrorSpy: jest.SpyInstance;
beforeAll(() => {
  consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
});

afterAll(() => {
  consoleErrorSpy.mockRestore();
});

describe('ScenarioPickerModal', () => {
  const mockOnClose = jest.fn();
  const mockOnSelectScenario = jest.fn();

  const mockCuratedScenarios: LoadableScenario[] = [
    createMockScenario('curated-1', 'Ring Placement Tutorial', 'basics'),
    createMockScenario('curated-2', 'Movement Basics', 'basics'),
  ];

  const mockVectorScenarios: LoadableScenario[] = [
    createMockScenario('vector-1', 'Capture Test', 'capture'),
    createMockScenario('vector-2', 'Line Formation Test', 'line'),
  ];

  const mockCustomScenarios: LoadableScenario[] = [
    createMockScenario('custom-1', 'My Saved Game', 'custom'),
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    mockLoadCuratedScenarios.mockResolvedValue(mockCuratedScenarios);
    mockLoadVectorScenarios.mockResolvedValue(mockVectorScenarios);
    mockLoadCustomScenarios.mockReturnValue(mockCustomScenarios);
    mockFilterScenarios.mockImplementation((scenarios) => scenarios);
  });

  const renderOpenModal = async (
    props: Partial<ScenarioPickerModalProps> = {},
    options?: { waitForLoad?: boolean }
  ) => {
    const waitForLoad = options?.waitForLoad ?? true;
    let utils: ReturnType<typeof render> | undefined;

    await act(async () => {
      utils = render(
        <ScenarioPickerModal
          isOpen={true}
          onClose={mockOnClose}
          onSelectScenario={mockOnSelectScenario}
          {...props}
        />
      );
    });

    if (waitForLoad) {
      await waitFor(() => expect(mockLoadCuratedScenarios).toHaveBeenCalled());
      await waitFor(() => expect(mockLoadVectorScenarios).toHaveBeenCalled());
      await waitFor(() => expect(screen.getByRole('dialog')).toBeInTheDocument());
    }

    return { user: userEvent.setup(), ...(utils as ReturnType<typeof render>) };
  };

  describe('visibility', () => {
    it('should not render when isOpen is false', () => {
      render(
        <ScenarioPickerModal
          isOpen={false}
          onClose={mockOnClose}
          onSelectScenario={mockOnSelectScenario}
        />
      );

      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('should render when isOpen is true', async () => {
      await renderOpenModal();
      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });
  });

  describe('header and controls', () => {
    it('should display the title', async () => {
      await renderOpenModal();
      expect(screen.getByText('Load Scenario')).toBeInTheDocument();
    });

    it('should have close button', async () => {
      await renderOpenModal();
      expect(screen.getByRole('button', { name: 'Close' })).toBeInTheDocument();
    });

    it('should call onClose when close button is clicked', async () => {
      const { user } = await renderOpenModal();

      await user.click(screen.getByRole('button', { name: 'Close' }));
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('tabs', () => {
    it('should display all three tabs', async () => {
      await renderOpenModal();

      await waitFor(() => {
        expect(screen.getByText(/Learning \(Rules \/ FAQ\)/)).toBeInTheDocument();
        expect(screen.getByText(/Test Scenarios/)).toBeInTheDocument();
        expect(screen.getByText(/My Scenarios/)).toBeInTheDocument();
      });
    });

    it('should display scenario counts in tabs', async () => {
      await renderOpenModal();

      await waitFor(() => {
        // The count appears in the tab text - check for the full tab text
        const curatedTab = screen.getByText(/Learning \(Rules \/ FAQ\)/);
        expect(curatedTab.closest('button')).toHaveTextContent('(2)');
      });
    });

    it('should switch tabs when clicked', async () => {
      const { user } = await renderOpenModal();

      await user.click(screen.getByText(/Test Scenarios/));

      // Vector scenarios should now be visible
      await waitFor(() => {
        expect(screen.getByText('Capture Test')).toBeInTheDocument();
      });
    });
  });

  describe('loading state', () => {
    it('should show loading state while scenarios load', async () => {
      // Make loading slow
      mockLoadCuratedScenarios.mockImplementation(() => new Promise(() => {}));

      await renderOpenModal({}, { waitForLoad: false });

      expect(screen.getByText('Loading scenarios...')).toBeInTheDocument();
    });
  });

  describe('scenario list', () => {
    it('should display curated scenarios by default', async () => {
      await renderOpenModal();

      await waitFor(() => {
        expect(screen.getByText('Ring Placement Tutorial')).toBeInTheDocument();
        expect(screen.getByText('Movement Basics')).toBeInTheDocument();
      });
    });

    it('should display scenario description', async () => {
      await renderOpenModal();

      await waitFor(() => {
        expect(screen.getByText('Description for Ring Placement Tutorial')).toBeInTheDocument();
      });
    });

    it('should display Load button for each scenario', async () => {
      await renderOpenModal();

      await waitFor(() => {
        const loadButtons = screen.getAllByRole('button', { name: 'Load' });
        expect(loadButtons.length).toBe(2);
      });
    });

    it('should call onSelectScenario when Load is clicked', async () => {
      const { user } = await renderOpenModal();

      const loadButtons = screen.getAllByRole('button', { name: 'Load' });
      await user.click(loadButtons[0]);

      expect(mockOnSelectScenario).toHaveBeenCalledWith(mockCuratedScenarios[0]);
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('search and filtering', () => {
    it('should display search input', async () => {
      await renderOpenModal();
      expect(screen.getByPlaceholderText('Search scenarios...')).toBeInTheDocument();
    });

    it('should display category filter', async () => {
      await renderOpenModal();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
      expect(screen.getByText('All Categories')).toBeInTheDocument();
    });

    it('should call filterScenarios when search changes', async () => {
      const { user } = await renderOpenModal();

      const searchInput = screen.getByPlaceholderText('Search scenarios...');
      await user.type(searchInput, 'ring');

      await waitFor(() => {
        expect(mockFilterScenarios).toHaveBeenCalled();
      });
    });
  });

  describe('custom scenarios tab', () => {
    it('should display custom scenarios', async () => {
      const { user } = await renderOpenModal();

      await user.click(screen.getByText(/My Scenarios/));

      await waitFor(() => {
        expect(screen.getByText('My Saved Game')).toBeInTheDocument();
      });
    });

    it('should show Import JSON button on custom tab', async () => {
      const { user } = await renderOpenModal({ developerToolsEnabled: true });

      await user.click(screen.getByText(/My Scenarios/));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Import JSON' })).toBeInTheDocument();
      });
    });

    it('should show Delete button for custom scenarios', async () => {
      const { user } = await renderOpenModal({ developerToolsEnabled: true });

      await user.click(screen.getByText(/My Scenarios/));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Delete' })).toBeInTheDocument();
      });
    });

    it('should show Export button for custom scenarios', async () => {
      const { user } = await renderOpenModal({ developerToolsEnabled: true });

      await user.click(screen.getByText(/My Scenarios/));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Export' })).toBeInTheDocument();
      });
    });

    it('should call deleteCustomScenario when Delete is clicked', async () => {
      const { user } = await renderOpenModal({ developerToolsEnabled: true });

      await user.click(screen.getByText(/My Scenarios/));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Delete' })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: 'Delete' }));

      expect(mockDeleteCustomScenario).toHaveBeenCalledWith('custom-1');
    });

    it('should call exportScenarioToFile when Export is clicked', async () => {
      const { user } = await renderOpenModal({ developerToolsEnabled: true });

      await user.click(screen.getByText(/My Scenarios/));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Export' })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: 'Export' }));

      expect(mockExportScenarioToFile).toHaveBeenCalledWith(mockCustomScenarios[0]);
    });
  });

  describe('empty states', () => {
    it('should show empty message for custom tab when no custom scenarios', async () => {
      mockLoadCustomScenarios.mockReturnValue([]);
      const { user } = await renderOpenModal();

      await user.click(screen.getByText(/My Scenarios/));

      await waitFor(() => {
        expect(
          screen.getByText(/No saved scenarios yet. Save a game state or import a JSON file/)
        ).toBeInTheDocument();
      });
    });

    it('should show empty message for curated tab when no scenarios', async () => {
      mockLoadCuratedScenarios.mockResolvedValue([]);

      await renderOpenModal();

      await waitFor(() => {
        expect(screen.getByText('No curated scenarios available yet.')).toBeInTheDocument();
      });
    });
  });

  describe('error handling', () => {
    it('should show error when loading fails', async () => {
      mockLoadCuratedScenarios.mockRejectedValue(new Error('Network error'));

      await renderOpenModal();

      await waitFor(() => {
        expect(screen.getByText('Failed to load scenarios')).toBeInTheDocument();
      });
    });

    it('should allow dismissing error', async () => {
      mockLoadCuratedScenarios.mockRejectedValue(new Error('Network error'));

      const { user } = await renderOpenModal();

      await waitFor(() => {
        expect(screen.getByText('Failed to load scenarios')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Dismiss'));

      expect(screen.queryByText('Failed to load scenarios')).not.toBeInTheDocument();
    });
  });

  describe('keyboard interactions', () => {
    it('should close on Escape key press', async () => {
      await renderOpenModal();

      // The keydown handler is on the inner dialogRef div
      const dialogContent = screen.getByRole('dialog').querySelector('div')!;
      fireEvent.keyDown(dialogContent, { key: 'Escape' });

      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('accessibility', () => {
    it('should have aria-modal attribute', async () => {
      await renderOpenModal();

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-modal', 'true');
    });

    it('should have aria-labelledby pointing to title', async () => {
      await renderOpenModal();

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-labelledby', 'scenario-picker-title');

      const title = document.getElementById('scenario-picker-title');
      expect(title).toHaveTextContent('Load Scenario');
    });
  });
});
