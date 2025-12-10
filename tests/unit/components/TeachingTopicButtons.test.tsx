import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TeachingTopicButtons } from '../../../src/client/components/TeachingOverlay';

describe('TeachingTopicButtons', () => {
  it('emits topic selections for mechanics and victory buttons', () => {
    const onSelectTopic = jest.fn();

    render(<TeachingTopicButtons onSelectTopic={onSelectTopic} />);

    fireEvent.click(screen.getByRole('button', { name: /Movement/i }));
    const territoryButtons = screen.getAllByRole('button', { name: /Territory/i });
    expect(territoryButtons).toHaveLength(2);
    fireEvent.click(territoryButtons[0]); // mechanics territory
    fireEvent.click(territoryButtons[1]); // victory territory
    fireEvent.click(screen.getByRole('button', { name: /Stalemate/i }));

    expect(onSelectTopic).toHaveBeenNthCalledWith(1, 'stack_movement');
    expect(onSelectTopic).toHaveBeenNthCalledWith(2, 'territory');
    expect(onSelectTopic).toHaveBeenNthCalledWith(3, 'victory_territory');
    expect(onSelectTopic).toHaveBeenNthCalledWith(4, 'victory_stalemate');
  });
});
