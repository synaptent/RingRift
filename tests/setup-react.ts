/**
 * React-specific Jest setup kept separate so non-React lanes can avoid
 * importing @testing-library/react and its MessagePort handle.
 */

import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});
