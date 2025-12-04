import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { SandboxTouchControlsPanel } from '../../src/client/components/SandboxTouchControlsPanel';
import type { Position } from '../../src/shared/types/game';

describe('SandboxTouchControlsPanel', () => {
  const basePosition: Position = { x: 3, y: 4 };

  const baseProps = {
    selectedPosition: undefined as Position | undefined,
    selectedStackDetails: null as {
      height: number;
      cap: number;
      controllingPlayer: number;
    } | null,
    validTargets: [] as Position[],
    isCaptureDirectionPending: false,
    captureTargets: [] as Position[],
    canUndoSegment: false,
    onClearSelection: jest.fn(),
    onUndoSegment: jest.fn(),
    onApplyMove: jest.fn(),
    showMovementGrid: true,
    onToggleMovementGrid: jest.fn(),
    showValidTargets: true,
    onToggleValidTargets: jest.fn(),
    phaseLabel: 'Ring Placement',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders basic touch controls UI', () => {
    render(<SandboxTouchControlsPanel {...baseProps} />);

    expect(screen.getByText(/Touch Controls/i)).toBeInTheDocument();
    expect(screen.getByText(/Phase: Ring Placement/i)).toBeInTheDocument();
    expect(screen.getByText(/Targets:/i)).toBeInTheDocument();
  });

  it('shows selection details when a position is selected', () => {
    const props = {
      ...baseProps,
      selectedPosition: basePosition,
      selectedStackDetails: {
        height: 2,
        cap: 1,
        controllingPlayer: 1,
      },
    };

    render(<SandboxTouchControlsPanel {...props} />);

    expect(screen.getByText('(3, 4)')).toBeInTheDocument();
    expect(screen.getByText(/H2 · C1 · P1/)).toBeInTheDocument();
  });

  it('invokes onClearSelection when enabled and button clicked', () => {
    const props = {
      ...baseProps,
      selectedPosition: basePosition,
    };

    render(<SandboxTouchControlsPanel {...props} />);

    const clearButton = screen.getByRole('button', { name: /Clear selection/i });
    expect(clearButton).not.toBeDisabled();

    fireEvent.click(clearButton);
    expect(baseProps.onClearSelection).toHaveBeenCalledTimes(1);
  });

  it('disables Undo last segment button when canUndoSegment=false', () => {
    render(<SandboxTouchControlsPanel {...baseProps} />);

    const undoButton = screen.getByRole('button', { name: /Undo last segment/i });
    expect(undoButton).toBeDisabled();
  });

  it('toggles movement grid and valid targets flags', () => {
    const props = {
      ...baseProps,
      onToggleMovementGrid: jest.fn(),
      onToggleValidTargets: jest.fn(),
    };

    render(<SandboxTouchControlsPanel {...props} />);

    const validTargetsCheckbox = screen.getByLabelText(/Show valid targets/i);
    const movementGridCheckbox = screen.getByLabelText(/Show movement grid/i);

    // uncheck valid targets
    fireEvent.click(validTargetsCheckbox);
    expect(props.onToggleValidTargets).toHaveBeenCalledWith(false);

    // uncheck movement grid
    fireEvent.click(movementGridCheckbox);
    expect(props.onToggleMovementGrid).toHaveBeenCalledWith(false);
  });

  describe('game storage UI', () => {
    it('does not render game storage section when props are undefined', () => {
      render(<SandboxTouchControlsPanel {...baseProps} />);

      expect(screen.queryByText(/Game storage/i)).not.toBeInTheDocument();
      expect(screen.queryByLabelText(/Auto-save completed games/i)).not.toBeInTheDocument();
    });

    it('renders game storage section when auto-save props are provided', () => {
      const props = {
        ...baseProps,
        autoSaveGames: true,
        onToggleAutoSave: jest.fn(),
        gameSaveStatus: 'idle' as const,
      };

      render(<SandboxTouchControlsPanel {...props} />);

      expect(screen.getByText(/Game storage/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Auto-save completed games/i)).toBeInTheDocument();
      expect(screen.getByText(/Stores finished games to the replay database/i)).toBeInTheDocument();
    });

    it('toggles auto-save when checkbox is clicked', () => {
      const onToggleAutoSave = jest.fn();
      const props = {
        ...baseProps,
        autoSaveGames: true,
        onToggleAutoSave,
        gameSaveStatus: 'idle' as const,
      };

      render(<SandboxTouchControlsPanel {...props} />);

      const checkbox = screen.getByLabelText(/Auto-save completed games/i);
      fireEvent.click(checkbox);

      expect(onToggleAutoSave).toHaveBeenCalledWith(false);
    });

    it('shows "Saving..." status indicator when gameSaveStatus is saving', () => {
      const props = {
        ...baseProps,
        autoSaveGames: true,
        onToggleAutoSave: jest.fn(),
        gameSaveStatus: 'saving' as const,
      };

      render(<SandboxTouchControlsPanel {...props} />);

      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });

    it('shows "Saved" status indicator when gameSaveStatus is saved', () => {
      const props = {
        ...baseProps,
        autoSaveGames: true,
        onToggleAutoSave: jest.fn(),
        gameSaveStatus: 'saved' as const,
      };

      render(<SandboxTouchControlsPanel {...props} />);

      expect(screen.getByText('Saved')).toBeInTheDocument();
    });

    it('shows "Error" status indicator when gameSaveStatus is error', () => {
      const props = {
        ...baseProps,
        autoSaveGames: true,
        onToggleAutoSave: jest.fn(),
        gameSaveStatus: 'error' as const,
      };

      render(<SandboxTouchControlsPanel {...props} />);

      expect(screen.getByText('Error')).toBeInTheDocument();
    });

    it('does not show status indicator when gameSaveStatus is idle', () => {
      const props = {
        ...baseProps,
        autoSaveGames: true,
        onToggleAutoSave: jest.fn(),
        gameSaveStatus: 'idle' as const,
      };

      render(<SandboxTouchControlsPanel {...props} />);

      expect(screen.queryByText('Saving...')).not.toBeInTheDocument();
      expect(screen.queryByText('Saved')).not.toBeInTheDocument();
      expect(screen.queryByText('Error')).not.toBeInTheDocument();
    });
  });

  describe('debug overlays UI', () => {
    it('does not render debug overlays section when toggle props are undefined', () => {
      render(<SandboxTouchControlsPanel {...baseProps} />);

      expect(screen.queryByText(/Debug overlays/i)).not.toBeInTheDocument();
      expect(screen.queryByLabelText(/Show detected lines/i)).not.toBeInTheDocument();
      expect(screen.queryByLabelText(/Show territory regions/i)).not.toBeInTheDocument();
    });

    it('renders debug overlays section when toggle props are provided', () => {
      const props = {
        ...baseProps,
        showLineOverlays: true,
        onToggleLineOverlays: jest.fn(),
        showTerritoryOverlays: true,
        onToggleTerritoryOverlays: jest.fn(),
      };

      render(<SandboxTouchControlsPanel {...props} />);

      expect(screen.getByText(/Debug overlays/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Show detected lines/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Show territory regions/i)).toBeInTheDocument();
      expect(
        screen.getByText(/Highlights completed lines and territory control/i)
      ).toBeInTheDocument();
    });

    it('toggles line overlays when checkbox is clicked', () => {
      const onToggleLineOverlays = jest.fn();
      const props = {
        ...baseProps,
        showLineOverlays: true,
        onToggleLineOverlays,
        showTerritoryOverlays: true,
        onToggleTerritoryOverlays: jest.fn(),
      };

      render(<SandboxTouchControlsPanel {...props} />);

      const checkbox = screen.getByLabelText(/Show detected lines/i);
      fireEvent.click(checkbox);

      expect(onToggleLineOverlays).toHaveBeenCalledWith(false);
    });

    it('toggles territory overlays when checkbox is clicked', () => {
      const onToggleTerritoryOverlays = jest.fn();
      const props = {
        ...baseProps,
        showLineOverlays: true,
        onToggleLineOverlays: jest.fn(),
        showTerritoryOverlays: true,
        onToggleTerritoryOverlays,
      };

      render(<SandboxTouchControlsPanel {...props} />);

      const checkbox = screen.getByLabelText(/Show territory regions/i);
      fireEvent.click(checkbox);

      expect(onToggleTerritoryOverlays).toHaveBeenCalledWith(false);
    });

    it('reflects current checkbox state from props', () => {
      const props = {
        ...baseProps,
        showLineOverlays: false,
        onToggleLineOverlays: jest.fn(),
        showTerritoryOverlays: true,
        onToggleTerritoryOverlays: jest.fn(),
      };

      render(<SandboxTouchControlsPanel {...props} />);

      const lineCheckbox = screen.getByLabelText(/Show detected lines/i) as HTMLInputElement;
      const territoryCheckbox = screen.getByLabelText(
        /Show territory regions/i
      ) as HTMLInputElement;

      expect(lineCheckbox.checked).toBe(false);
      expect(territoryCheckbox.checked).toBe(true);
    });
  });
});
