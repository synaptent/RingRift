import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { GameFilters } from '../../../src/client/components/ReplayPanel/GameFilters';
import type { ReplayGameQueryParams } from '../../../src/client/types/replay';

describe('GameFilters', () => {
  const defaultProps = {
    filters: {} as ReplayGameQueryParams,
    onFilterChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders board type filter', () => {
      render(<GameFilters {...defaultProps} />);

      expect(screen.getByLabelText(/Filter by board type/i)).toBeInTheDocument();
    });

    it('renders player count filter', () => {
      render(<GameFilters {...defaultProps} />);

      expect(screen.getByLabelText(/Filter by player count/i)).toBeInTheDocument();
    });

    it('renders outcome filter', () => {
      render(<GameFilters {...defaultProps} />);

      expect(screen.getByLabelText(/Filter by outcome/i)).toBeInTheDocument();
    });

    it('renders source filter', () => {
      render(<GameFilters {...defaultProps} />);

      expect(screen.getByLabelText(/Filter by source/i)).toBeInTheDocument();
    });
  });

  describe('board type options', () => {
    it('has All Boards option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by board type/i);
      expect(select).toContainHTML('<option value="">All Boards</option>');
    });

    it('has 8×8 option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by board type/i);
      expect(select).toContainHTML('<option value="square8">8×8</option>');
    });

    it('has 19×19 option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by board type/i);
      expect(select).toContainHTML('<option value="square19">19×19</option>');
    });

    it('has Hex option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by board type/i);
      expect(select).toContainHTML('<option value="hexagonal">Hex</option>');
    });
  });

  describe('player count options', () => {
    it('has Any Players option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by player count/i);
      expect(select).toContainHTML('<option value="">Any Players</option>');
    });

    it('has 2 Players option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by player count/i);
      expect(select).toContainHTML('<option value="2">2 Players</option>');
    });

    it('has 3 Players option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by player count/i);
      expect(select).toContainHTML('<option value="3">3 Players</option>');
    });

    it('has 4 Players option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by player count/i);
      expect(select).toContainHTML('<option value="4">4 Players</option>');
    });
  });

  describe('termination options', () => {
    it('has Any Outcome option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by outcome/i);
      expect(select).toContainHTML('<option value="">Any Outcome</option>');
    });

    it('has Ring Elim. option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by outcome/i);
      expect(select).toContainHTML('<option value="ring_elimination">Ring Elim.</option>');
    });

    it('has Territory option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by outcome/i);
      expect(select).toContainHTML('<option value="territory">Territory</option>');
    });

    it('has Last Standing option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by outcome/i);
      expect(select).toContainHTML('<option value="last_player_standing">Last Standing</option>');
    });

    it('has Stalemate option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by outcome/i);
      expect(select).toContainHTML('<option value="stalemate">Stalemate</option>');
    });
  });

  describe('source options', () => {
    it('has Any Source option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by source/i);
      expect(select).toContainHTML('<option value="">Any Source</option>');
    });

    it('has Self-play option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by source/i);
      expect(select).toContainHTML('<option value="self_play">Self-play</option>');
    });

    it('has Sandbox option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by source/i);
      expect(select).toContainHTML('<option value="sandbox">Sandbox</option>');
    });

    it('has Tournament option', () => {
      render(<GameFilters {...defaultProps} />);

      const select = screen.getByLabelText(/Filter by source/i);
      expect(select).toContainHTML('<option value="tournament">Tournament</option>');
    });
  });

  describe('current filter values', () => {
    it('shows current board type selection', () => {
      render(<GameFilters {...defaultProps} filters={{ board_type: 'square8' }} />);

      expect(screen.getByLabelText(/Filter by board type/i)).toHaveValue('square8');
    });

    it('shows current player count selection', () => {
      render(<GameFilters {...defaultProps} filters={{ num_players: 3 }} />);

      expect(screen.getByLabelText(/Filter by player count/i)).toHaveValue('3');
    });

    it('shows current outcome selection', () => {
      render(<GameFilters {...defaultProps} filters={{ termination_reason: 'territory' }} />);

      expect(screen.getByLabelText(/Filter by outcome/i)).toHaveValue('territory');
    });

    it('shows current source selection', () => {
      render(<GameFilters {...defaultProps} filters={{ source: 'self_play' }} />);

      expect(screen.getByLabelText(/Filter by source/i)).toHaveValue('self_play');
    });

    it('shows empty value for unset filters', () => {
      render(<GameFilters {...defaultProps} filters={{}} />);

      expect(screen.getByLabelText(/Filter by board type/i)).toHaveValue('');
      expect(screen.getByLabelText(/Filter by player count/i)).toHaveValue('');
      expect(screen.getByLabelText(/Filter by outcome/i)).toHaveValue('');
      expect(screen.getByLabelText(/Filter by source/i)).toHaveValue('');
    });
  });

  describe('filter change interactions', () => {
    it('calls onFilterChange when board type changes', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(<GameFilters {...defaultProps} onFilterChange={onFilterChange} />);

      await user.selectOptions(screen.getByLabelText(/Filter by board type/i), 'square8');

      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({ board_type: 'square8', offset: 0 })
      );
    });

    it('calls onFilterChange when player count changes', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(<GameFilters {...defaultProps} onFilterChange={onFilterChange} />);

      await user.selectOptions(screen.getByLabelText(/Filter by player count/i), '2');

      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({ num_players: 2, offset: 0 })
      );
    });

    it('calls onFilterChange when outcome changes', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(<GameFilters {...defaultProps} onFilterChange={onFilterChange} />);

      await user.selectOptions(screen.getByLabelText(/Filter by outcome/i), 'ring_elimination');

      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({ termination_reason: 'ring_elimination', offset: 0 })
      );
    });

    it('calls onFilterChange when source changes', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(<GameFilters {...defaultProps} onFilterChange={onFilterChange} />);

      await user.selectOptions(screen.getByLabelText(/Filter by source/i), 'sandbox');

      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({ source: 'sandbox', offset: 0 })
      );
    });
  });

  describe('filter clearing', () => {
    it('removes board_type when All Boards is selected', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(
        <GameFilters
          {...defaultProps}
          filters={{ board_type: 'square8' }}
          onFilterChange={onFilterChange}
        />
      );

      await user.selectOptions(screen.getByLabelText(/Filter by board type/i), '');

      const calledFilters = onFilterChange.mock.calls[0][0];
      expect(calledFilters.board_type).toBeUndefined();
    });

    it('removes num_players when Any Players is selected', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(
        <GameFilters
          {...defaultProps}
          filters={{ num_players: 2 }}
          onFilterChange={onFilterChange}
        />
      );

      await user.selectOptions(screen.getByLabelText(/Filter by player count/i), '');

      const calledFilters = onFilterChange.mock.calls[0][0];
      expect(calledFilters.num_players).toBeUndefined();
    });

    it('removes termination_reason when Any Outcome is selected', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(
        <GameFilters
          {...defaultProps}
          filters={{ termination_reason: 'territory' }}
          onFilterChange={onFilterChange}
        />
      );

      await user.selectOptions(screen.getByLabelText(/Filter by outcome/i), '');

      const calledFilters = onFilterChange.mock.calls[0][0];
      expect(calledFilters.termination_reason).toBeUndefined();
    });

    it('removes source when Any Source is selected', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(
        <GameFilters
          {...defaultProps}
          filters={{ source: 'self_play' }}
          onFilterChange={onFilterChange}
        />
      );

      await user.selectOptions(screen.getByLabelText(/Filter by source/i), '');

      const calledFilters = onFilterChange.mock.calls[0][0];
      expect(calledFilters.source).toBeUndefined();
    });
  });

  describe('offset reset', () => {
    it('resets offset to 0 when filter changes', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(
        <GameFilters {...defaultProps} filters={{ offset: 20 }} onFilterChange={onFilterChange} />
      );

      await user.selectOptions(screen.getByLabelText(/Filter by board type/i), 'hexagonal');

      expect(onFilterChange).toHaveBeenCalledWith(expect.objectContaining({ offset: 0 }));
    });
  });

  describe('preserving other filters', () => {
    it('preserves existing filters when changing another', async () => {
      const onFilterChange = jest.fn();
      const user = userEvent.setup();
      render(
        <GameFilters
          {...defaultProps}
          filters={{ board_type: 'square8', num_players: 2 }}
          onFilterChange={onFilterChange}
        />
      );

      await user.selectOptions(screen.getByLabelText(/Filter by source/i), 'self_play');

      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({
          board_type: 'square8',
          num_players: 2,
          source: 'self_play',
          offset: 0,
        })
      );
    });
  });

  describe('custom className', () => {
    it('applies custom className to container', () => {
      const { container } = render(<GameFilters {...defaultProps} className="custom-class" />);

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });

  describe('accessibility', () => {
    it('all selects have accessible labels', () => {
      render(<GameFilters {...defaultProps} />);

      expect(screen.getByLabelText(/Filter by board type/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Filter by player count/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Filter by outcome/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Filter by source/i)).toBeInTheDocument();
    });
  });
});
