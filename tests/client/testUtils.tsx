import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { render } from '@testing-library/react';

/**
 * Render a React element wrapped in a MemoryRouter, useful for page and host
 * components that expect a router context.
 */
export function renderWithRouter(
  ui: React.ReactElement,
  { route = '/' }: { route?: string } = {}
) {
  return render(<MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>);
}