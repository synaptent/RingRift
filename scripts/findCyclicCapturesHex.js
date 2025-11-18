// Search for cyclic overtaking capture sequences on a hex board
// (RingRift hexagonal board) looking for sequences where a set of N
// target stacks are overtaken in a multi-segment chain, and at least
// two of them are overtaken more than once during the sequence.
//
// This mirrors scripts/findCyclicCaptures.js but uses hex geometry:
// - Board: hex with cube coords, radius R (default 10 for size 11).
// - Movement directions: 6 cube directions.
// - capHeight == stackHeight for all stacks (captures always allowed
//   as long as target still has rings and capHeight condition holds).
// - We explicitly model markers and collapsed spaces per compact rules.
// - We assume a single moving player (P) and no opponent stacks.
//
// The script is designed to iteratively search over increasing
// depth limits and window radii (within reasonable bounds), so it
// will keep exploring more generous configurations until either:
//   - a sequence is found, or
//   - all configured (depth, radius) combinations are exhausted.
//
// Parameters (via env):
//   NUM_TARGETS_LIST    e.g. "2,3,4" (default: 2,3,4)
//   DEPTH_LIST          e.g. "6,8,10" (default: 6,8,10)
//   HEX_RADIUS          board radius (default 10)
//   WINDOW_RADIUS_LIST  e.g. "3,5,7" (default: 3,5,7)
//   MAX_COMBOS_PER_N    limit on combinations per (N, depth, radius)
//   MAX_STATES          soft cap on explored states per run (for pruning)

// CONFIGURATION
// -------------
const HEX_RADIUS = parseInt(process.env.HEX_RADIUS || '10', 10); // RingRift hex11

const DEFAULT_NUM_TARGETS_LIST = '2,3,4';
const numTargetsListEnv = process.env.NUM_TARGETS_LIST || DEFAULT_NUM_TARGETS_LIST;
const NUM_TARGETS_LIST = numTargetsListEnv
  .split(',')
  .map((s) => parseInt(s.trim(), 10))
  .filter((n) => Number.isFinite(n) && n >= 2 && n <= 6);

const depthListEnv = process.env.DEPTH_LIST || '6,8,10';
const DEPTH_LIST = depthListEnv
  .split(',')
  .map((s) => parseInt(s.trim(), 10))
  .filter((n) => Number.isFinite(n) && n >= 4 && n <= 12);

// Attacker starting position at cube origin.
const ATTACKER_START = { x: 0, y: 0, z: 0 };

// Radius around ATTACKER_START for candidate target positions (hex distance).
const windowRadiusListEnv = process.env.WINDOW_RADIUS_LIST || '3,5,7';
const WINDOW_RADIUS_LIST = windowRadiusListEnv
  .split(',')
  .map((s) => parseInt(s.trim(), 10))
  .filter((n) => Number.isFinite(n) && n >= 1 && n <= HEX_RADIUS);

const MAX_COMBOS_PER_N = parseInt(process.env.MAX_COMBOS_PER_N || '20000', 10);
const MAX_STATES = parseInt(process.env.MAX_STATES || '200000', 10);

// Cube directions (6 neighbors).
const HEX_DIRS = [
  { dx: 1, dy: -1, dz: 0 },
  { dx: 1, dy: 0, dz: -1 },
  { dx: 0, dy: 1, dz: -1 },
  { dx: -1, dy: 1, dz: 0 },
  { dx: -1, dy: 0, dz: 1 },
  { dx: 0, dy: -1, dz: 1 },
];

// HELPERS
// -------

function hexDist(a, b) {
  return (
    Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z)
  ) / 2;
}

function hexInBounds(x, y, z) {
  return x + y + z === 0 && Math.max(Math.abs(x), Math.abs(y), Math.abs(z)) <= HEX_RADIUS;
}

function key(x, y, z) {
  return `${x},${y},${z}`;
}

function cloneState(state) {
  return {
    attacker: { ...state.attacker },
    targets: state.targets.map((t) => ({ ...t })),
    markers: { ...state.markers },
    collapsed: new Set(Array.from(state.collapsed)),
  };
}

function getTargetAt(state, x, y, z) {
  return (
    state.targets.find((t) => t.alive && t.x === x && t.y === y && t.z === z) || null
  );
}

function isCollapsed(state, x, y, z) {
  return state.collapsed.has(key(x, y, z));
}

// Generate legal capture segments from the current attacker position.
// For hex, distance is hexDist; path cells must avoid stacks & collapsed.
function generateSegments(state) {
  const segs = [];
  const { attacker } = state;
  const H = attacker.height;

  for (const dir of HEX_DIRS) {
    let x = attacker.x;
    let y = attacker.y;
    let z = attacker.z;

    // Walk along the ray until we hit a target, a collapsed cell, or out-of-bounds.
    for (let step = 1; step <= 2 * HEX_RADIUS; step++) {
      x += dir.dx;
      y += dir.dy;
      z += dir.dz;
      if (!hexInBounds(x, y, z)) break;
      if (isCollapsed(state, x, y, z)) break;

      const t = getTargetAt(state, x, y, z);
      if (t) {
        const tx = t.x;
        const ty = t.y;
        const tz = t.z;

        // Look beyond for landings along same direction.
        let lx = tx;
        let ly = ty;
        let lz = tz;

        for (let step2 = step + 1; step2 <= 2 * HEX_RADIUS; step2++) {
          lx += dir.dx;
          ly += dir.dy;
          lz += dir.dz;
          if (!hexInBounds(lx, ly, lz)) break;
          if (isCollapsed(state, lx, ly, lz)) break;

          // cannot land on another stack
          if (getTargetAt(state, lx, ly, lz)) break;

          const dist = hexDist(attacker, { x: lx, y: ly, z: lz });
          if (dist >= H) {
            segs.push({
              from: { x: attacker.x, y: attacker.y, z: attacker.z },
              dir,
              target: { name: t.name, x: tx, y: ty, z: tz },
              landing: { x: lx, y: ly, z: lz },
              dist,
            });
          }
        }
        break; // stop scanning this ray after first target
      }
    }
  }
  return segs;
}

// Marker & collapse processing along a hex path.
function processMarkersAlongPath(state, seg) {
  const { from, target, landing, dir } = seg;
  const { markers, collapsed } = state;

  const fromKey = key(from.x, from.y, from.z);
  if (!collapsed.has(fromKey)) {
    markers[fromKey] = 'P';
  }

  function processCell(x, y, z) {
    const k = key(x, y, z);
    if (collapsed.has(k)) return false;
    const owner = markers[k];
    if (owner === 'P') {
      collapsed.add(k);
      delete markers[k];
    } else if (owner === 'Q') {
      markers[k] = 'P';
    }
    return true;
  }

  // from → target
  let x = from.x;
  let y = from.y;
  let z = from.z;
  while (true) {
    x += dir.dx;
    y += dir.dy;
    z += dir.dz;
    if (x === target.x && y === target.y && z === target.z) break;
    if (!processCell(x, y, z)) return false;
  }

  // target → landing
  x = target.x;
  y = target.y;
  z = target.z;
  while (true) {
    x += dir.dx;
    y += dir.dy;
    z += dir.dz;
    if (x === landing.x && y === landing.y && z === landing.z) break;
    if (!processCell(x, y, z)) return false;
  }

  const landingKey = key(landing.x, landing.y, landing.z);
  if (collapsed.has(landingKey)) return false;
  if (markers[landingKey] === 'P') delete markers[landingKey];

  return true;
}

function applySegment(state, seg) {
  const next = cloneState(state);
  const tgt = getTargetAt(next, seg.target.x, seg.target.y, seg.target.z);
  if (!tgt) return null;

  if (next.attacker.capHeight < tgt.capHeight) return null;
  if (!processMarkersAlongPath(next, seg)) return null;

  next.attacker.x = seg.landing.x;
  next.attacker.y = seg.landing.y;
  next.attacker.z = seg.landing.z;
  next.attacker.height += 1;
  next.attacker.capHeight = next.attacker.height;

  tgt.height -= 1;
  tgt.capHeight = Math.max(0, tgt.capHeight - 1);
  if (tgt.height <= 0) tgt.alive = false;

  return next;
}

// Coarse state key for pruning.
function stateKey(state) {
  const a = state.attacker;
  const attackerPart = `${a.x},${a.y},${a.z},${a.height}`;
  const targetsPart = state.targets
    .map((t) =>
      t.alive
        ? `${t.name}:${t.x},${t.y},${t.z},${t.height}`
        : `${t.name}:dead`,
    )
    .join('|');
  const collapsedPart = Array.from(state.collapsed).sort().join(';');
  return `${attackerPart}|${targetsPart}|${collapsedPart}`;
}

let found = false;
let initialTargets = [];
let visited = new Map();
let statesExplored = 0;

function successPredicate(history) {
  const counts = {};
  for (const h of history) {
    counts[h.targetName] = (counts[h.targetName] || 0) + 1;
  }
  const names = initialTargets.map((t) => t.name);
  if (!names.every((name) => (counts[name] || 0) >= 1)) return false;
  const numAtLeastTwo = names.filter((name) => (counts[name] || 0) >= 2).length;
  return numAtLeastTwo >= 2;
}

function search(state, history, depthLimit) {
  if (found) return;
  if (statesExplored >= MAX_STATES) return;

  const remainingDepth = depthLimit - history.length;
  const keyStr = stateKey(state);
  const prev = visited.get(keyStr);
  if (prev !== undefined && prev >= remainingDepth) return;
  visited.set(keyStr, remainingDepth);

  statesExplored++;

  if (history.length >= depthLimit) {
    if (successPredicate(history)) {
      found = true;
      console.log('FOUND SEQUENCE (hex):');
      console.log('Initial targets:', initialTargets);
      console.log('History:');
      for (const h of history) console.log(h);
    }
    return;
  }

  const segs = generateSegments(state);
  if (segs.length === 0) return;

  // Optional heuristic: explore segments in order of descending distance
  // to favour longer jumps first.
  segs.sort((a, b) => b.dist - a.dist);

  for (const seg of segs) {
    const next = applySegment(state, seg);
    if (!next) continue;
    search(
      next,
      history.concat({
        from: seg.from,
        targetName: seg.target.name,
        targetPos: { x: seg.target.x, y: seg.target.y, z: seg.target.z },
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

// Generate candidate cube coords within a given WINDOW_RADIUS of ATTACKER_START.
function generateCoordsWindow(windowRadius) {
  const coords = [];
  for (let dx = -windowRadius; dx <= windowRadius; dx++) {
    for (let dy = -windowRadius; dy <= windowRadius; dy++) {
      const dz = -dx - dy;
      const x = ATTACKER_START.x + dx;
      const y = ATTACKER_START.y + dy;
      const z = ATTACKER_START.z + dz;
      if (!hexInBounds(x, y, z)) continue;
      const d = hexDist(ATTACKER_START, { x, y, z });
      if (d > 0 && d <= windowRadius) {
        coords.push([x, y, z]);
      }
    }
  }
  return coords;
}

function runForNumTargets(numTargets, depthLimit, windowRadius) {
  found = false;
  visited = new Map();
  statesExplored = 0;

  const coords = generateCoordsWindow(windowRadius);
  const allCombos = combinations(coords, numTargets);
  const limitedCombos = allCombos.slice(0, MAX_COMBOS_PER_N);

  console.log(
    `\n=== HEX N=${numTargets}, depth=${depthLimit}, windowRadius=${windowRadius}, combos=${limitedCombos.length} ===`,
  );

  const targetNames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').slice(0, numTargets);

  for (const combo of limitedCombos) {
    if (
      combo.some(
        ([x, y, z]) => x === ATTACKER_START.x && y === ATTACKER_START.y && z === ATTACKER_START.z,
      )
    ) {
      continue;
    }

    const targets = combo.map(([x, y, z], idx) => ({
      name: targetNames[idx],
      x,
      y,
      z,
      height: 4,
      capHeight: 4,
      alive: true,
    }));

    const initial = {
      attacker: { ...ATTACKER_START, height: 2, capHeight: 2 },
      targets,
      markers: {},
      collapsed: new Set(),
    };

    initialTargets = initial.targets.map((t) => ({ name: t.name }));

    search(initial, [], depthLimit);
    if (found) {
      console.log(
        'Result: sequence FOUND for (N, depth, windowRadius) =',
        numTargets,
        depthLimit,
        windowRadius,
      );
      console.log('States explored:', statesExplored);
      return true;
    }
  }

  console.log(
    'Result: no sequence found for (N, depth, windowRadius) =',
    numTargets,
    depthLimit,
    windowRadius,
    'states explored =',
    statesExplored,
  );
  return false;
}

function main() {
  if (NUM_TARGETS_LIST.length === 0) {
    console.log('NUM_TARGETS_LIST is empty; nothing to do.');
    return;
  }

  console.log('findCyclicCapturesHex hex search (iterative)');
  console.log('NUM_TARGETS_LIST =', NUM_TARGETS_LIST);
  console.log('DEPTH_LIST =', DEPTH_LIST);
  console.log('HEX_RADIUS =', HEX_RADIUS);
  console.log('WINDOW_RADIUS_LIST =', WINDOW_RADIUS_LIST);
  console.log('MAX_COMBOS_PER_N =', MAX_COMBOS_PER_N);
  console.log('MAX_STATES =', MAX_STATES);

  for (const n of NUM_TARGETS_LIST) {
    let foundForN = false;
    for (const depth of DEPTH_LIST) {
      for (const r of WINDOW_RADIUS_LIST) {
        const ok = runForNumTargets(n, depth, r);
        if (ok) {
          foundForN = true;
          break;
        }
      }
      if (foundForN) break;
    }
    if (!foundForN) {
      console.log(
        `No sequences found for N=${n} within all configured depths and windows; ` +
          'this does not prove impossibility but suggests such patterns are rare within these bounds.',
      );
    }
  }
}

main();
