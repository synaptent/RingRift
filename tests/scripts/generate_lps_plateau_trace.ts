import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Move,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../../src/shared/types/game';
import {
  computeProgressSnapshot,
  hashGameState,
  countRingsInPlayForPlayer,
} from '../../src/shared/engine/core';
import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';

type FixtureVersion = 'v2';

interface TraceStepExpected {
  tsValid?: boolean;
  tsStateHash?: string;
  tsS?: number;
}

interface TraceStep {
  label?: string;
  move: Move;
  expected?: TraceStepExpected;
  stateHash?: string;
  sInvariant?: number;
}

interface TraceFixture {
  version: FixtureVersion;
  boardType: BoardType;
  initialState: GameState;
  steps: TraceStep[];
  meta?: {
    seed: number;
    plateauActionIndex: number;
    lpsCandidate: number;
  };
}

function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function createThreePlayerSandbox(boardType: BoardType): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType,
    numPlayers: 3,
    playerKinds: ['ai', 'ai', 'ai'],
  };

  const handler: SandboxInteractionHandler = {
    async requestChoice<TChoice extends PlayerChoice>(
      choice: TChoice
    ): Promise<PlayerChoiceResponseFor<TChoice>> {
      const anyChoice = choice as any;

      if (anyChoice.type === 'capture_direction') {
        const cd = anyChoice as CaptureDirectionChoice;
        const options = cd.options ?? [];
        if (options.length === 0) {
          throw new Error('LPS harness: no options for capture_direction');
        }

        // Deterministic tie-breaker: pick option with smallest landing x,y.
        let selected = options[0];
        for (const opt of options) {
          if (
            opt.landingPosition.x < selected.landingPosition.x ||
            (opt.landingPosition.x === selected.landingPosition.x &&
              opt.landingPosition.y < selected.landingPosition.y)
          ) {
            selected = opt;
          }
        }

        return {
          choiceId: cd.id,
          playerNumber: cd.playerNumber,
          choiceType: cd.type,
          selectedOption: selected,
        } as PlayerChoiceResponseFor<TChoice>;
      }

      const firstOption = (anyChoice.options && anyChoice.options[0]) ?? undefined;
      return {
        choiceId: anyChoice.id,
        playerNumber: anyChoice.playerNumber,
        choiceType: anyChoice.type,
        selectedOption: firstOption,
      } as PlayerChoiceResponseFor<TChoice>;
    },
  };

  return new ClientSandboxEngine({ config, interactionHandler: handler });
}

/**
 * Convert a live GameState with Map-based board collections into a
 * JSON-serialisable shape expected by Python parity fixtures.
 */
function toFixtureGameState(state: GameState): GameState {
  const boardAny = state.board as any;

  const stacks: Record<string, unknown> = {};
  boardAny.stacks.forEach((value: unknown, key: string) => {
    stacks[key] = value;
  });

  const markers: Record<string, unknown> = {};
  boardAny.markers.forEach((value: unknown, key: string) => {
    markers[key] = value;
  });

  const collapsedSpaces: Record<string, unknown> = {};
  boardAny.collapsedSpaces.forEach((value: unknown, key: string) => {
    collapsedSpaces[key] = value;
  });

  const territories: Record<string, unknown> = {};
  boardAny.territories.forEach((value: unknown, key: string) => {
    territories[key] = value;
  });

  return {
    ...(state as any),
    board: {
      ...boardAny,
      stacks,
      markers,
      collapsedSpaces,
      territories,
    },
  } as GameState;
}

class LpsTracker {
  private readonly ruleEngine: RuleEngine;
  private roundIndex = 0;
  private currentRoundFirstPlayer: number | null = null;
  private currentRoundActorMask: Map<number, boolean> = new Map();
  private exclusivePlayerForCompletedRound: number | null = null;

  constructor(boardType: BoardType) {
    const boardManager = new BoardManager(boardType);
    this.ruleEngine = new RuleEngine(boardManager, boardType);
  }

  private hasAnyRealActionForPlayer(state: GameState, playerNumber: number): boolean {
    if (state.gameStatus !== 'active') {
      return false;
    }

    const playerState = state.players.find((p) => p.playerNumber === playerNumber);
    if (!playerState) {
      return false;
    }

    // 1) Ring placement.
    if (playerState.ringsInHand > 0) {
      const placementState: GameState = {
        ...state,
        currentPlayer: playerNumber,
        currentPhase: 'ring_placement',
      };
      const placementMoves = this.ruleEngine.getValidMoves(placementState);
      if (placementMoves.some((m) => m.type === 'place_ring')) {
        return true;
      }
    }

    // 2) Non-capture movement.
    const movementState: GameState = {
      ...state,
      currentPlayer: playerNumber,
      currentPhase: 'movement',
    };
    const movementMoves = this.ruleEngine.getValidMoves(movementState);
    if (
      movementMoves.some(
        (m) => m.type === 'move_stack' || m.type === 'move_stack' || m.type === 'build_stack'
      )
    ) {
      return true;
    }

    // 3) Overtaking capture.
    const captureState: GameState = {
      ...state,
      currentPlayer: playerNumber,
      currentPhase: 'capture',
    };
    const captureMoves = this.ruleEngine.getValidMoves(captureState);
    return captureMoves.some((m) => m.type === 'overtaking_capture');
  }

  private playerHasMaterial(state: GameState, playerNumber: number): boolean {
    const totalInPlay = countRingsInPlayForPlayer(state as any, playerNumber);
    return totalInPlay > 0;
  }

  public updateForStartOfTurn(state: GameState): void {
    if (state.gameStatus !== 'active') {
      return;
    }

    const phase = state.currentPhase;
    if (
      phase !== 'ring_placement' &&
      phase !== 'movement' &&
      phase !== 'capture' &&
      phase !== 'chain_capture'
    ) {
      return;
    }

    const currentPlayer = state.currentPlayer;

    if (this.currentRoundFirstPlayer === null || this.currentRoundActorMask.size === 0) {
      this.currentRoundFirstPlayer = currentPlayer;
      this.currentRoundActorMask = new Map();
    } else if (currentPlayer === this.currentRoundFirstPlayer) {
      this.finaliseCompletedRound();
      this.currentRoundFirstPlayer = currentPlayer;
      this.currentRoundActorMask = new Map();
    }

    const hasReal = this.hasAnyRealActionForPlayer(state, currentPlayer);
    this.currentRoundActorMask.set(currentPlayer, hasReal);
  }

  private finaliseCompletedRound(): void {
    if (this.currentRoundActorMask.size === 0) {
      this.exclusivePlayerForCompletedRound = null;
      return;
    }

    let candidate: number | null = null;
    for (const [playerNumber, hadReal] of this.currentRoundActorMask.entries()) {
      if (!hadReal) {
        continue;
      }
      if (candidate === null) {
        candidate = playerNumber;
      } else {
        candidate = null;
        break;
      }
    }

    this.exclusivePlayerForCompletedRound = candidate;
    this.roundIndex += 1;
  }

  /**
   * Return the LPS plateau candidate at the start of the current turn, if any.
   */
  public maybeGetPlateauCandidate(state: GameState): number | null {
    if (state.gameStatus !== 'active') {
      return null;
    }

    const phase = state.currentPhase;
    if (
      phase !== 'ring_placement' &&
      phase !== 'movement' &&
      phase !== 'capture' &&
      phase !== 'chain_capture'
    ) {
      return null;
    }

    const candidate = this.exclusivePlayerForCompletedRound;
    if (candidate == null || state.currentPlayer !== candidate) {
      return null;
    }

    if (!this.hasAnyRealActionForPlayer(state, candidate)) {
      this.exclusivePlayerForCompletedRound = null;
      return null;
    }

    const othersHaveRealActions = state.players.some((p) => {
      if (p.playerNumber === candidate) {
        return false;
      }
      if (!this.playerHasMaterial(state, p.playerNumber)) {
        return false;
      }
      return this.hasAnyRealActionForPlayer(state, p.playerNumber);
    });

    if (othersHaveRealActions) {
      this.exclusivePlayerForCompletedRound = null;
      return null;
    }

    return candidate;
  }
}

async function findFirstLpsPlateau(
  seed: number,
  maxActions: number
): Promise<{
  plateauState: GameState;
  plateauActionIndex: number;
  lpsCandidate: number;
  engine: ClientSandboxEngine;
} | null> {
  const rng = makePrng(seed);
  const originalRandom = Math.random;
  (Math as any).random = rng;

  try {
    const boardType: BoardType = 'square8';
    const engine = createThreePlayerSandbox(boardType);
    const tracker = new LpsTracker(boardType);

    let plateauState: GameState | null = null;
    let plateauActionIndex = 0;
    let lpsCandidate = 0;

    for (let actionIndex = 0; actionIndex < maxActions; actionIndex += 1) {
      const before = engine.getGameState();

      tracker.updateForStartOfTurn(before);
      const candidate = tracker.maybeGetPlateauCandidate(before);
      if (candidate != null) {
        plateauState = JSON.parse(JSON.stringify(before)) as GameState;
        plateauActionIndex = actionIndex;
        lpsCandidate = candidate;
        break;
      }

      if (before.gameStatus !== 'active') {
        break;
      }

      const current = before.players.find((p) => p.playerNumber === before.currentPlayer);
      if (!current || current.type !== 'ai') {
        break;
      }

      await engine.maybeRunAITurn();
    }

    if (!plateauState) {
      return null;
    }

    return { plateauState, plateauActionIndex, lpsCandidate, engine };
  } finally {
    (Math as any).random = originalRandom;
  }
}

function buildTraceFromPlateau(
  plateauState: GameState,
  _engine: ClientSandboxEngine,
  lpsCandidate: number,
  seed: number,
  plateauActionIndex: number
): TraceFixture {
  const fixtureState = toFixtureGameState(plateauState);
  const progress = computeProgressSnapshot(plateauState as any);
  const hash = hashGameState(plateauState as any);

  const fixture: TraceFixture = {
    version: 'v2',
    boardType: plateauState.boardType,
    initialState: fixtureState,
    steps: [],
    meta: {
      seed,
      plateauActionIndex,
      lpsCandidate,
    },
  };

  const pseudoMove: Move = {
    id: 'lps-plateau-noop',
    type: 'skip_placement',
    player: plateauState.currentPlayer,
    to: { x: 0, y: 0 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: plateauState.moveHistory.length + 1,
  };

  fixture.steps.push({
    label: 'lps_plateau_noop',
    move: pseudoMove,
    expected: {
      tsValid: false,
      tsStateHash: hash,
      tsS: progress.S,
    },
    stateHash: hash,
    sInvariant: progress.S,
  });

  return fixture;
}

async function main(): Promise<void> {
  const seed = Number(process.env.LPS_SEED ?? '1');
  const maxActions = Number(process.env.LPS_MAX_ACTIONS ?? '256');

  const result = await findFirstLpsPlateau(seed, maxActions);
  if (!result) {
    // eslint-disable-next-line no-console
    console.log(
      '[LPS harness] no LPS plateau found for seed=' + seed + ' within ' + maxActions + ' actions'
    );
    return;
  }

  const { plateauState, plateauActionIndex, lpsCandidate, engine } = result;

  const trace = buildTraceFromPlateau(plateauState, engine, lpsCandidate, seed, plateauActionIndex);

  const outDir = join(__dirname, '..', 'fixtures', 'rules-parity', 'v2');
  mkdirSync(outDir, { recursive: true });
  const outPath = join(outDir, 'trace.square8_3p.lps_plateau.json');
  writeFileSync(outPath, JSON.stringify(trace, null, 2), 'utf-8');

  // eslint-disable-next-line no-console
  console.log(
    '[LPS harness] wrote plateau trace to ' +
      outPath +
      ' (seed=' +
      seed +
      ', plateauActionIndex=' +
      plateauActionIndex +
      ', candidate=' +
      lpsCandidate +
      ')'
  );
}

main();
