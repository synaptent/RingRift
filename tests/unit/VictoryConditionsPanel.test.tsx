import React from 'react';
import { render, screen } from '@testing-library/react';
import { VictoryConditionsPanel } from '../../src/client/components/GameHUD';

describe('VictoryConditionsPanel', () => {
  it('renders ring and territory victory guidance', () => {
    render(<VictoryConditionsPanel />);

    expect(screen.getByTestId('victory-conditions-help')).toBeInTheDocument();
    expect(screen.getByText('Victory Conditions')).toBeInTheDocument();
    expect(screen.getByText(/Ring Elimination/i)).toBeInTheDocument();
    expect(screen.getByText(/Territory Control/i)).toBeInTheDocument();
    expect(screen.getByText(/Last Player Standing/i)).toBeInTheDocument();
  });
});
