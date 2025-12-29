# RingRift Game Record Specification

> **SSoT alignment:** The canonical schema is defined in
> `src/shared/types/gameRecord.ts` and mirrored in
> `ai-service/app/models/game_record.py`. If this document conflicts with
> those types, the code wins and this doc must be updated.
>
> **Notation:** `MoveRecord.rrn` (RingRift Notation) is defined in
> `src/shared/types/gameRecord.ts` (`moveRecordToRRN`, `legacyGameRecordToRRN`).
> This is separate from the ai-service algebraic notation in
> `ai-service/docs/specs/GAME_NOTATION_SPEC.md`.

## 1. Overview

A GameRecord is the canonical, JSONL-friendly representation of a **completed**
RingRift game. It is used for training pipelines, replay tooling, and historical
storage. Each record is self-contained and does **not** include full board
snapshots per move; instead, it stores a compact MoveRecord list.

Key traits:

- Canonical and versioned via `metadata.recordVersion`.
- Player numbers are **1-based** (match `GameState.currentPlayer`).
- Designed for JSONL export (`gameRecordToJsonlLine`).

## 2. Schema Summary

```json
{
  "id": "game-uuid",
  "boardType": "square8",
  "numPlayers": 2,
  "rngSeed": 12345,
  "isRated": true,
  "players": [
    {
      "playerNumber": 1,
      "username": "Alice",
      "playerType": "human",
      "ratingBefore": 1200,
      "ratingAfter": 1210
    },
    {
      "playerNumber": 2,
      "username": "Bot",
      "playerType": "ai",
      "aiDifficulty": 6,
      "aiType": "mcts"
    }
  ],
  "winner": 1,
  "outcome": "ring_elimination",
  "finalScore": {
    "ringsEliminated": { "1": 18, "2": 14 },
    "territorySpaces": { "1": 6, "2": 2 },
    "ringsRemaining": { "1": 0, "2": 4 }
  },
  "startedAt": "2025-12-30T01:02:03Z",
  "endedAt": "2025-12-30T01:07:45Z",
  "totalMoves": 47,
  "totalDurationMs": 342000,
  "moves": [
    {
      "moveNumber": 1,
      "player": 1,
      "type": "place_ring",
      "to": { "x": 3, "y": 3 },
      "thinkTimeMs": 1200,
      "rrn": "Pd4"
    }
  ],
  "metadata": {
    "recordVersion": "1.0",
    "createdAt": "2025-12-30T01:07:45Z",
    "source": "online_game",
    "tags": ["canonical"]
  },
  "initialStateHash": "...",
  "finalStateHash": "...",
  "progressSnapshots": []
}
```

## 3. Enums

### GameOutcome

```
ring_elimination
territory_control
last_player_standing
timeout
resignation
draw
abandonment
```

### RecordSource

```
online_game
self_play
cmaes_optimization
tournament
soak_test
manual_import
```

## 4. MoveRecord

MoveRecord is a compact move entry used in `moves[]`.

Required fields:

- `moveNumber` (1-based sequence number)
- `player` (playerNumber)
- `type` (MoveType)
- `thinkTimeMs`

Common optional fields:

- `from`, `to`
- `captureTarget`
- `placementCount`, `placedOnStack`
- `formedLines`, `collapsedMarkers`, `disconnectedRegions`, `eliminatedRings`
- `rrn` (RingRift Notation string)

Python-specific extension:

- `mctsPolicy` may appear in Python GameRecords as a dict of move-index →
  probability. TS consumers should ignore unknown fields.

## 5. Metadata

`metadata` carries record-level context:

- `recordVersion` (default "1.0")
- `createdAt` (timestamp)
- `source` / `sourceId`
- `generation`, `candidateId` (evolutionary workflows)
- `tags` (string list)
- `fsmValidated` (optional validation flag; Python only)

## 6. Notes

- GameRecord is for **completed** games; in-progress states belong in
  GameReplayDB or GameState snapshots.
- For replay DB schema, see `ai-service/docs/specs/GAME_REPLAY_DATABASE_SPEC.md`.
- For RRN encoding, see `src/shared/types/gameRecord.ts`.
