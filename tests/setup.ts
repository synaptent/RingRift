/**
 * Full Jest setup for the general test environment.
 *
 * Supported-path rules/parity lanes use tests/setup-core.ts directly to avoid
 * importing React Testing Library, which leaves a MessagePort open under React 19.
 */

import './setup-core';
import './setup-react';
