# Contract Test Vectors v2

This directory contains test vectors for contract-based parity testing between the TypeScript canonical engine and Python AI rules engine.

## Vector Format

All vectors follow the schema defined in `src/shared/engine/contracts/schemas.ts` (TestVectorSchema).

### Structure

```json
{
  "id": "category.scenario.variant",
  "version": "v2",
  "category": "placement|movement|capture|chain_capture|line_processing|territory_processing|victory|edge_case",
  "description": "Human-readable description",
  "tags": ["smoke", "regression", "edge-case"],
  "source": "manual|recorded|generated|regression",
  "createdAt": "ISO-8601 timestamp",
  "input": {
    "state": {
      /* SerializedGameState */
    },
    "move": {
      /* Move */
    }
  },
  "expectedOutput": {
    "status": "complete|awaiting_decision",
    "assertions": {
      "currentPlayer": 1,
      "currentPhase": "movement",
      "gameStatus": "active",
      "stackCount": 1,
      "markerCount": 0,
      "sInvariantDelta": 0
    }
  }
}
```

## Vector Files

| File                     | Category  | Description                                 |
| ------------------------ | --------- | ------------------------------------------- |
| `placement.vectors.json` | placement | Ring placement and skip placement scenarios |
| `movement.vectors.json`  | movement  | Stack movement scenarios                    |
| `capture.vectors.json`   | capture   | Overtaking capture scenarios                |

## Usage

### TypeScript

```typescript
import { importVectorBundle, validateAgainstAssertions } from '@/shared/engine/contracts';
import { readFileSync } from 'fs';

// Load vectors
const json = readFileSync('tests/fixtures/contract-vectors/v2/placement.vectors.json', 'utf-8');
const vectors = importVectorBundle(json);

// Run vector through engine
for (const vector of vectors) {
  const result = processTurn(deserializeGameState(vector.input.state), vector.input.move);
  const validation = validateAgainstAssertions(result.nextState, vector.expectedOutput.assertions);

  if (!validation.valid) {
    console.error(`Vector ${vector.id} failed:`, validation.failures);
  }
}
```

### Python

```python
import json
from pathlib import Path

# Load vectors
vectors_path = Path('tests/fixtures/contract-vectors/v2/placement.vectors.json')
with vectors_path.open() as f:
    bundle = json.load(f)

# Run vector through engine
for vector in bundle['vectors']:
    state = deserialize_game_state(vector['input']['state'])
    move = vector['input']['move']
    result = process_turn(state, move)

    # Validate assertions
    assertions = vector['expectedOutput']['assertions']
    assert result.current_player == assertions.get('currentPlayer')
    assert result.current_phase == assertions.get('currentPhase')
```

## Adding New Vectors

### Manual Creation

1. Create a new vector JSON object following the schema
2. Add to the appropriate category file or create a new file
3. Ensure the vector passes on TypeScript engine before committing

### Automated Generation

Use the test vector generator to create vectors from game traces:

```typescript
import { createContractTestVector, exportVectorBundle } from '@/shared/engine/contracts';

// Record before/after states for each move
const vector = createContractTestVector(beforeState, move, afterState, {
  description: 'Custom scenario description',
  tags: ['regression', 'bug-123'],
  source: 'regression',
});

// Export batch
const json = exportVectorBundle([vector, ...moreVectors]);
```

## Categories

| Category               | Description                             |
| ---------------------- | --------------------------------------- |
| `placement`            | Ring placement and skip_placement moves |
| `movement`             | Stack movement (move_stack, move_ring)  |
| `capture`              | Initial overtaking captures             |
| `chain_capture`        | Chain capture continuation segments     |
| `line_processing`      | Line detection and reward selection     |
| `territory_processing` | Territory regions and elimination       |
| `victory`              | Game-ending scenarios                   |
| `edge_case`            | Unusual or boundary scenarios           |

## Tags

- `smoke`: Core functionality tests (run on every commit)
- `regression`: Tests for specific bug fixes
- `edge-case`: Boundary condition tests
- `parity`: Tests focusing on TS↔Python alignment
- `hexagonal`: Hexagonal board variants

## Migration from v1

v2 vectors differ from v1 in:

1. **Schema**: Uses contract schema from `src/shared/engine/contracts/`
2. **Assertions**: Explicit assertions instead of full state comparison
3. **Status field**: Supports `awaiting_decision` for pending player choices
4. **S-Invariant tracking**: Always includes `sInvariantDelta` assertion
