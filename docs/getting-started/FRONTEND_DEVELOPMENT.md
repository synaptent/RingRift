# Frontend Development Setup Guide

This guide covers setting up and developing the RingRift frontend (React/TypeScript client).

## Prerequisites

- **Node.js**: v18.0.0 or higher (v20+ recommended)
- **npm**: v9.0.0 or higher
- **PostgreSQL**: For backend database (optional for frontend-only work)
- **Redis**: For session caching (optional for frontend-only work)

## Quick Start

```bash
# Install dependencies
npm install

# Start frontend only (Vite dev server)
npm run dev:client

# Start both frontend and backend
npm run dev

# Start sandbox diagnostics mode (for debugging AI/game issues)
npm run dev:sandbox:diagnostics
```

## Development Scripts

| Script                            | Description                                      |
| --------------------------------- | ------------------------------------------------ |
| `npm run dev:client`              | Start Vite dev server (frontend only, port 5173) |
| `npm run dev:server`              | Start Node.js server (backend only, port 3000)   |
| `npm run dev`                     | Start both frontend and backend concurrently     |
| `npm run dev:sandbox:diagnostics` | Start sandbox with diagnostic logging enabled    |
| `npm run dev:free-port`           | Kill process on port 3000                        |
| `npm run dev:free-all`            | Kill processes on ports 3000, 3001, 5173         |
| `npm run dev:app`                 | Free port 3000 then start full dev environment   |
| `npm run dev:doctor`              | Run diagnostic checks on dev environment         |

## Project Structure

```
src/client/
├── adapters/          # Data transformation layers (game state → view models)
├── components/        # React UI components
├── contexts/          # React context providers (GameContext, AuthContext)
├── facades/           # High-level hooks combining multiple data sources
├── hooks/             # Custom React hooks
├── pages/             # Page-level components (routes)
├── sandbox/           # Client-side game engine for offline/sandbox play
├── services/          # API clients and external service integrations
├── styles/            # CSS and Tailwind configurations
└── utils/             # Utility functions and helpers
```

## Key Concepts

### Game Modes

1. **Backend Mode**: Game state managed by server, real-time multiplayer via WebSocket
2. **Sandbox Mode**: Game state managed client-side, for local play and AI testing

### View Models

The `gameViewModels.ts` adapter transforms raw game state into view-ready models:

- `BoardViewModel`: Cell positions, highlights, interactions
- `PlayerViewModel`: Player info, scores, turn status
- `PendingChoiceViewModel`: Current decision state for player

### Pending Choices

When players need to make decisions (line order, territory region, ring elimination), the game enters a "pending choice" state. Components check `pendingChoice` and render appropriate UI.

## Debugging Tools

### Freeze Debugger

The `freezeDebugger.ts` utility captures game state before AI turns to diagnose browser freezes.

**Enable in browser console:**

```javascript
window.__FREEZE_DEBUGGER__.enable();
```

**After a freeze, open a new tab and run:**

```javascript
// View last captured state
window.__FREEZE_DEBUGGER__.getLastState();

// Export state as JSON file
window.__FREEZE_DEBUGGER__.exportState();

// View state history
window.__FREEZE_DEBUGGER__.getHistory();

// Export all history
window.__FREEZE_DEBUGGER__.exportHistory();

// Get stats
window.__FREEZE_DEBUGGER__.getStats();

// Clear stored data
window.__FREEZE_DEBUGGER__.clearData();
```

**How it works:**

1. Saves state to localStorage before each AI turn
2. Watchdog timer warns after 2s, critical alert after 5s
3. Stores last 10 states in memory, 100 in IndexedDB
4. State persists across browser restarts

### React Query DevTools

React Query DevTools are available in development for inspecting query cache, mutations, and refetch behavior.

### Browser DevTools

- **React DevTools**: Component tree, props, state inspection
- **Network tab**: WebSocket frames for real-time communication
- **Performance tab**: Identify rendering bottlenecks

## Common Workflows

### Adding a New Component

1. Create component in `src/client/components/`
2. Add styles using Tailwind classes or CSS modules
3. Write tests in `tests/unit/components/`
4. Update snapshot tests if needed

### Modifying Game State Display

1. Update view model in `src/client/adapters/gameViewModels.ts`
2. Update consuming components
3. Run `npm run test:fast` to verify

### Debugging AI Issues

1. Enable freeze debugger: `__FREEZE_DEBUGGER__.enable()`
2. Play game until issue occurs
3. Export state: `__FREEZE_DEBUGGER__.exportState()`
4. Share JSON file for analysis

### Testing

```bash
# Run fast unit tests (components, shared logic)
npm run test:fast

# Run all unit tests
npm run test:unit

# Run with coverage
npm run test:coverage:components

# Watch mode
npm run test:watch
```

## Environment Variables

Create `.env` file in project root:

```bash
# API endpoints (defaults shown)
VITE_API_URL=http://localhost:3000
VITE_WS_URL=ws://localhost:3000

# Feature flags
VITE_ENABLE_DEV_TOOLS=true
```

## Troubleshooting

### Port Already in Use

```bash
npm run dev:free-all
npm run dev
```

### TypeScript Errors

```bash
npx tsc --noEmit
```

### Lint Issues

```bash
npm run lint        # Check for issues
npm run lint:fix    # Auto-fix issues
```

### Tests Hanging

```bash
npm run test:kill-jest
npm run test:fast
```

## Related Documentation

- [Architecture Overview](../ARCHITECTURE_OVERVIEW.md)
- [State Machines](../architecture/STATE_MACHINES.md)
- [Phase Orchestration](../architecture/PHASE_ORCHESTRATION_ARCHITECTURE.md)
