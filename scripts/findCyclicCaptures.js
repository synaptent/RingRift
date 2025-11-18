// Search for cyclic overtaking capture sequences on a square board
// looking for sequences where a set of N target stacks are overtaken
// in a multi-segment chain, and at least two of them are overtaken
// more than once during the sequence.
//
// This search models a simplified subset of the RingRift compact rules:
// - Board: square19, movement directions are the 8 Moore directions.
// - capHeight == stackHeight for all stacks (captures always allowed
//   as long as target still has rings and capHeight condition holds).
// - We explicitly model markers and collapsed spaces according to
//   Sections 3 & 4 of the compact rules:
//     * Leaving a stack drops a marker for the moving player (P).
//     * Passing over your own marker collapses that cell (permanent
//       obstacle for stacks and markers).
//     * Passing over an opponent marker flips it to your color.
//     * Landing on your own marker removes it (no collapse).
//     * Collapsed spaces block movement/capture paths and cannot
//       hold stacks or markers.
// - We assume a single moving player (P) and no opponent stacks.
//   Opponent markers are not seeded initially.
//
// The goal is to empirically explore:
//   - N = 2 targets: whether both can realistically be overtaken
//     multiple times in a single chain under full path/marker rules.
//   - N = 3 and N = 4 targets: whether such patterns become easier
//     to realise on a large board.
//
// The script is parameterised via environment variables so you can run:
//   NUM_TARGETS_LIST="2,3,4" DEPTH_LIMIT=6 WINDOW_RADIUS=2 node scripts/findCyclicCaptures.js
// and adjust as needed.

const BOARD_SIZE = 19; // square19

// CONFIGURABLE PARAMETERS (via env)
// ---------------------------------
const DEFAULT_NUM_TARGETS_LIST = '2,3,4';
const numTargetsListEnv = process.env.NUM_TARGETS_LIST || DEFAULT_NUM_TARGETS_LIST;
const NUM_TARGETS_LIST = numTargetsListEnv
  .split(',')
  .map((s) => parseInt(s.trim(), 10))
  .filter((n) => Number.isFinite(n) && n >= 2 && n <= 4);

const DEPTH_LIMIT = parseInt(process.env.DEPTH_LIMIT || '6', 10);

// Attacker starting position (near centre of 19x19).
const ATTACKER_START = { x: 9, y: 9 };

// Radius around ATTACKER_START for candidate target positions.
// Effective window is (2 * WINDOW_RADIUS + 1)^2 cells.
const WINDOW_RADIUS = parseInt(process.env.WINDOW_RADIUS || '2', 10);

// Limit how many target-combination seeds we explore per NUM_TARGETS
// to keep runtime manageable. If there are more combinations than this,
// we just take the first MAX_COMBOS_PER_N (deterministically).
const MAX_COMBOS_PER_N = parseInt(process.env.MAX_COMBOS_PER_N || '20000', 10);

// ---------------------------------

const dirs = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [-1, -1],
  [1, -1],
  [-1, 1],
];

function inBounds(x, y) {
  return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE;
}

function key(x, y) {
  return `${x},${y}`;
}

function cloneState(state) {
  return {
    attacker: { ...state.attacker },
    targets: state.targets.map((t) => ({ ...t })),
    markers: { ...state.markers },
    collapsed: new Set(Array.from(state.collapsed)),
  };
}

function getTargetAt(state, x, y) {
  return state.targets.find((t) => t.alive && t.x === x && t.y === y) || null;
}

function isCollapsed(state, x, y) {
  return state.collapsed.has(key(x, y));
}

// Generate legal capture segments from the current attacker position,
// respecting stacks and collapsed spaces. Markers on the path are
// allowed; their effects are handled when applying the segment.
function generateSegments(state) {
  const segs = [];
  const { attacker } = state;
  const H = attacker.height;

  for (const [dx, dy] of dirs) {
    let x = attacker.x;
    let y = attacker.y;

    // Walk along the ray until we either hit a target, a collapsed
    // cell, or leave the board.
    for (let step = 1; step < BOARD_SIZE; step++) {
      x += dx;
      y += dy;
      if (!inBounds(x, y)) break;
      if (isCollapsed(state, x, y)) break;

      const t = getTargetAt(state, x, y);
      if (t) {
        const tx = t.x;
        const ty = t.y;

        // Now look beyond for landings. Along target→landing path,
        // we must avoid stacks and collapsed cells.
        let lx = tx;
        let ly = ty;

        for (let step2 = step + 1; step2 < BOARD_SIZE; step2++) {
          lx += dx;
          ly += dy;
          if (!inBounds(lx, ly)) break;
          if (isCollapsed(state, lx, ly)) break;

          // cannot land on another stack
          if (getTargetAt(state, lx, ly)) break;

          const dist = step2; // path length along this ray from from→landing
          if (dist >= H) {
            segs.push({
              from: { x: attacker.x, y: attacker.y },
              dir: { dx, dy },
              target: { name: t.name, x: tx, y: ty },
              landing: { x: lx, y: ly },
              dist,
            });
          }
        }
        break; // stop scanning this ray after the first target
      }
    }
  }
  return segs;
}

// Process markers and collapsed spaces along the path of a capture
// segment. Returns false if the path is blocked by a collapsed space.
function processMarkersAlongPath(state, seg) {
  const { from, target, landing, dir } = seg;
  const { markers, collapsed } = state;
  const dx = dir.dx;
  const dy = dir.dy;

  // At departure: leave a marker for P.
  const fromKey = key(from.x, from.y);
  if (!collapsed.has(fromKey)) {
    markers[fromKey] = 'P';
  }

  // Helper to process a single intermediate cell.
  function processCell(x, y) {
    const k = key(x, y);
    if (collapsed.has(k)) {
      return false; // path already blocked
    }
    const owner = markers[k];
    if (owner === 'P') {
      // Crossing own marker collapses this cell.
      collapsed.add(k);
      delete markers[k];
    } else if (owner === 'Q') {
      // Crossing opponent marker flips it.
      markers[k] = 'P';
    }
    return true;
  }

  // from → target (excluding endpoints)
  let x = from.x;
  let y = from.y;
  while (true) {
    x += dx;
    y += dy;
    if (x === target.x && y === target.y) break;
    if (!processCell(x, y)) return false;
  }

  // target → landing (excluding endpoints)
  x = target.x;
  y = target.y;
  while (true) {
    x += dx;
    y += dy;
    if (x === landing.x && y === landing.y) break;
    if (!processCell(x, y)) return false;
  }

  // Landing: if there's a P marker, remove it (no collapse).
  const landingKey = key(landing.x, landing.y);
  if (collapsed.has(landingKey)) return false;
  if (markers[landingKey] === 'P') {
    delete markers[landingKey];
  }

  return true;
}

function applySegment(state, seg) {
  const next = cloneState(state);
  const tgt = getTargetAt(next, seg.target.x, seg.target.y);
  if (!tgt) return null;

  // Cap condition: attacker capHeight >= target capHeight
  if (next.attacker.capHeight < tgt.capHeight) return null;

  // Process markers/collapses along path. If path blocked, segment is invalid.
  if (!processMarkersAlongPath(next, seg)) return null;

  // Update attacker position and height
  next.attacker.x = seg.landing.x;
  next.attacker.y = seg.landing.y;
  next.attacker.height += 1;
  next.attacker.capHeight = next.attacker.height;

  // Remove one ring from target
  tgt.height -= 1;
  tgt.capHeight = Math.max(0, tgt.capHeight - 1);
  if (tgt.height <= 0) {
    tgt.alive = false;
  }

  return next;
}

// Coarse state key for pruning: attacker pos + height, targets' alive/height,
// and collapsed-space pattern. If we revisit the same key with **greater or
// equal remaining depth**, we can prune.
function stateKey(state) {
  const a = state.attacker;
  const attackerPart = `${a.x},${a.y},${a.height}`;
  const targetsPart = state.targets
    .map((t) => (t.alive ? `${t.name}:${t.x},${t.y},${t.height}` : `${t.name}:dead`))
    .join('|');
  const collapsedPart = Array.from(state.collapsed).sort().join(';');
  return `${attackerPart}|${targetsPart}|${collapsedPart}`;
}

let found = false;
let initialTargets = [];
let visited = new Map(); // key -> maxRemainingDepth seen

function successPredicate(history) {
  // Count how many times each target name was hit.
  const counts = {};
  for (const h of history) {
    counts[h.targetName] = (counts[h.targetName] || 0) + 1;
  }

  const names = initialTargets.map((t) => t.name);

  // All initial targets must be hit at least once
  if (!names.every((name) => (counts[name] || 0) >= 1)) return false;

  // At least two distinct targets must be hit at least twice
  const numAtLeastTwo = names.filter((name) => (counts[name] || 0) >= 2).length;
  return numAtLeastTwo >= 2;
}

function search(state, history, depthLimit) {
  if (found) return;

  const remainingDepth = depthLimit - history.length;
  const keyStr = stateKey(state);
  const prev = visited.get(keyStr);
  if (prev !== undefined && prev >= remainingDepth) {
    return; // already explored this state with >= remaining depth
  }
  visited.set(keyStr, remainingDepth);

  if (history.length >= depthLimit) {
    if (successPredicate(history)) {
      found = true;
      console.log('FOUND SEQUENCE:');
      console.log('Initial targets:', initialTargets);
      console.log('History:');
      for (const h of history) {
        console.log(h);
      }
    }
    return;
  }

  const segs = generateSegments(state);
  if (segs.length === 0) return;

  for (const seg of segs) {
    const next = applySegment(state, seg);
    if (!next) continue;
    search(
      next,
      history.concat({
        from: seg.from,
        targetName: seg.target.name,
        targetPos: { x: seg.target.x, y: seg.target.y },
        landing: seg.landing,
        dist: seg.dist,
        attackerHeightAfter: next.attacker.height,
      }),
      depthLimit,
    );
    if (found) return;
  }
}

function combinations(arr, k, start = 0, acc = [], out = []) {
  if (acc.length === k) {
    out.push(acc.slice());
    return out;
  }
  for (let i = start; i < arr.length; i++) {
    acc.push(arr[i]);
    combinations(arr, k, i + 1, acc, out);
    acc.pop();
  }
  return out;
}

function generateCoordsWindow() {
  const coords = [];
  const minX = Math.max(0, ATTACKER_START.x - WINDOW_RADIUS);
  const maxX = Math.min(BOARD_SIZE - 1, ATTACKER_START.x + WINDOW_RADIUS);
  const minY = Math.max(0, ATTACKER_START.y - WINDOW_RADIUS);
  const maxY = Math.min(BOARD_SIZE - 1, ATTACKER_START.y + WINDOW_RADIUS);
  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      coords.push([x, y]);
    }
  }
  return coords;
}

function runForNumTargets(numTargets) {
  found = false;
  visited = new Map();

  const coords = generateCoordsWindow();
  const allCombos = combinations(coords, numTargets);
  const limitedCombos = allCombos.slice(0, MAX_COMBOS_PER_N);

  console.log(
    `\n=== NUM_TARGETS = ${numTargets}, combos considered = ${limitedCombos.length}, DEPTH_LIMIT = ${DEPTH_LIMIT}, WINDOW_RADIUS = ${WINDOW_RADIUS} ===`,
  );

  // Name targets A, B, C,...
  const targetNames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').slice(0, numTargets);

  for (const combo of limitedCombos) {
    // Skip if any target overlaps attacker start
    if (combo.some(([x, y]) => x === ATTACKER_START.x && y === ATTACKER_START.y)) {
      continue;
    }

    const targets = combo.map(([x, y], idx) => ({
      name: targetNames[idx],
      x,
      y,
      height: 4,
      capHeight: 4,
      alive: true,
    }));

    const initial = {
      attacker: {
        x: ATTACKER_START.x,
        y: ATTACKER_START.y,
        height: 2,
        capHeight: 2,
      },
      targets,
      markers: {},
      collapsed: new Set(),
    };

    initialTargets = initial.targets.map((t) => ({ name: t.name }));

    search(initial, [], DEPTH_LIMIT);
    if (found) {
      console.log('Result: sequence found for NUM_TARGETS =', numTargets);
      return;
    }
  }

  console.log('Result: no sequence found for NUM_TARGETS =', numTargets);
}

function main() {
  if (NUM_TARGETS_LIST.length === 0) {
    console.log('NUM_TARGETS_LIST is empty; nothing to do.');
    return;
  }

  console.log('findCyclicCaptures square19 search');
  console.log('NUM_TARGETS_LIST =', NUM_TARGETS_LIST);
  console.log('DEPTH_LIMIT =', DEPTH_LIMIT);
  console.log('WINDOW_RADIUS =', WINDOW_RADIUS);
  console.log('MAX_COMBOS_PER_N =', MAX_COMBOS_PER_N);

  for (const n of NUM_TARGETS_LIST) {
    runForNumTargets(n);
  }
}

main();
