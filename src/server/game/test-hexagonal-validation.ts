/**
 * Hexagonal Board Validation Test
 *
 * Purpose: Validate that the hexagonal board implementation satisfies all requirements:
 * 1. Generates exactly 331 positions for size=11 hexagonal board
 * 2. All positions satisfy cube coordinate constraint (x + y + z = 0)
 * 3. Line detection uses exactly 3 axes (not 4 like square boards)
 * 4. All adjacency operations use 6-direction hexagonal
 * 5. Movement validation handles hexagonal distances correctly
 * 6. Territory disconnection uses hexagonal adjacency
 */

import { BoardManager } from './BoardManager';
import { BoardType, Position, positionToString } from '../../shared/types/game';

console.log('=== HEXAGONAL BOARD VALIDATION TEST ===\n');

// Test 1: Position Generation
console.log('TEST 1: Position Generation');
console.log('----------------------------');

const hexManager = new BoardManager('hexagonal' as BoardType);
// Initialize the board for completeness; no need to keep a reference.
hexManager.createBoard();
const hexPositions = hexManager.getAllPositions();

console.log(`Expected positions: 331`);
console.log(`Actual positions: ${hexPositions.length}`);
console.log(`✓ Position count: ${hexPositions.length === 331 ? 'PASS' : 'FAIL'}\n`);

// Test 2: Cube Coordinate Constraint (x + y + z = 0)
console.log('TEST 2: Cube Coordinate Constraint');
console.log('-----------------------------------');

let coordinateViolations = 0;
const sampleViolations: Position[] = [];

for (const pos of hexPositions) {
  const sum = pos.x + pos.y + (pos.z || 0);
  if (sum !== 0) {
    coordinateViolations++;
    if (sampleViolations.length < 5) {
      sampleViolations.push(pos);
    }
  }
}

console.log(`Positions checked: ${hexPositions.length}`);
console.log(`Violations found: ${coordinateViolations}`);
if (sampleViolations.length > 0) {
  console.log('Sample violations:');
  sampleViolations.forEach((pos) => {
    console.log(
      `  Position ${positionToString(pos)}: x=${pos.x}, y=${pos.y}, z=${pos.z}, sum=${pos.x + pos.y + (pos.z || 0)}`
    );
  });
}
console.log(`✓ Cube coordinates: ${coordinateViolations === 0 ? 'PASS' : 'FAIL'}\n`);

// Test 3: Line Directions (Should be 3, not 4)
console.log('TEST 3: Line Detection Axes');
console.log('----------------------------');

// Create a test position and check line directions
const centerPos: Position = { x: 0, y: 0, z: 0 };
hexManager['boardType'] = 'hexagonal';

// Access private method via type assertion
const lineDirections = (hexManager as any).getLineDirections();

console.log(`Expected line axes: 3`);
console.log(`Actual line axes: ${lineDirections.length}`);
console.log('Line directions:');
lineDirections.forEach((dir: Position, idx: number) => {
  console.log(`  Axis ${idx + 1}: (${dir.x}, ${dir.y}, ${dir.z || 0})`);
});
console.log(`✓ Line axes count: ${lineDirections.length === 3 ? 'PASS' : 'FAIL'}\n`);

// Test 4: Hexagonal Adjacency (6 neighbors)
console.log('TEST 4: Hexagonal Adjacency');
console.log('---------------------------');

const neighbors = hexManager.getNeighbors(centerPos, 'hexagonal');
console.log(`Expected neighbors: 6`);
console.log(`Actual neighbors: ${neighbors.length}`);
console.log('Neighbor positions from (0,0,0):');
neighbors.forEach((n, idx) => {
  console.log(`  ${idx + 1}. (${n.x}, ${n.y}, ${n.z || 0})`);
});
console.log(`✓ Neighbor count: ${neighbors.length === 6 ? 'PASS' : 'FAIL'}\n`);

// Test 5: All Adjacency Types Use Hexagonal
console.log('TEST 5: Adjacency Type Configuration');
console.log('-------------------------------------');

const config = hexManager.getConfig();
console.log(`Movement adjacency: ${config.movementAdjacency}`);
console.log(`Line adjacency: ${config.lineAdjacency}`);
console.log(`Territory adjacency: ${config.territoryAdjacency}`);

const allHexagonal =
  config.movementAdjacency === 'hexagonal' &&
  config.lineAdjacency === 'hexagonal' &&
  config.territoryAdjacency === 'hexagonal';

console.log(`✓ All adjacency types hexagonal: ${allHexagonal ? 'PASS' : 'FAIL'}\n`);

// Test 6: Hexagonal Distance Calculation
console.log('TEST 6: Hexagonal Distance Calculation');
console.log('---------------------------------------');

const pos1: Position = { x: 0, y: 0, z: 0 };
const pos2: Position = { x: 2, y: -1, z: -1 };
const distance = hexManager.calculateDistance(pos1, pos2);

// Hexagonal distance should be max(|dx|, |dy|, |dz|)
const expectedDistance = Math.max(
  Math.abs(pos2.x - pos1.x),
  Math.abs(pos2.y - pos1.y),
  Math.abs((pos2.z || 0) - (pos1.z || 0))
);

console.log(`Position 1: ${positionToString(pos1)}`);
console.log(`Position 2: ${positionToString(pos2)}`);
console.log(`Expected distance: ${expectedDistance}`);
console.log(`Actual distance: ${distance}`);
console.log(`✓ Distance calculation: ${distance === expectedDistance ? 'PASS' : 'FAIL'}\n`);

// Test 7: Edge Detection
console.log('TEST 7: Edge Position Detection');
console.log('--------------------------------');

const edgePositions = hexManager.getEdgePositions();
console.log(`Edge positions found: ${edgePositions.length}`);

// For a hexagonal board with size=11, radius is 10 (size - 1)
// Edge positions should be at distance = radius
const edgeRadius = 10;
let incorrectEdges = 0;
for (const pos of edgePositions) {
  const dist = Math.max(Math.abs(pos.x), Math.abs(pos.y), Math.abs(pos.z || 0));
  if (dist !== edgeRadius) {
    incorrectEdges++;
  }
}

console.log(`Expected edge radius: ${edgeRadius}`);
console.log(`Incorrect edge positions: ${incorrectEdges}`);
console.log(`✓ Edge detection: ${incorrectEdges === 0 ? 'PASS' : 'FAIL'}\n`);

// Test 8: Compare with Square Boards
console.log('TEST 8: Comparison with Square Boards');
console.log('--------------------------------------');

const square8Manager = new BoardManager('square8' as BoardType);
const square8Config = square8Manager.getConfig();
const square8LineDirections = (square8Manager as any).getLineDirections();

console.log('Square 8x8 board:');
console.log(`  Line axes: ${square8LineDirections.length} (should be 4)`);
console.log(`  Movement adjacency: ${square8Config.movementAdjacency} (should be moore)`);
console.log(`  Territory adjacency: ${square8Config.territoryAdjacency} (should be von_neumann)`);
console.log();

console.log('Hexagonal board:');
console.log(`  Line axes: ${lineDirections.length} (should be 3)`);
console.log(`  Movement adjacency: ${config.movementAdjacency} (should be hexagonal)`);
console.log(`  Territory adjacency: ${config.territoryAdjacency} (should be hexagonal)`);
console.log();

const differencesCorrect =
  square8LineDirections.length === 4 &&
  lineDirections.length === 3 &&
  square8Config.territoryAdjacency === 'von_neumann' &&
  config.territoryAdjacency === 'hexagonal';

console.log(`✓ Board type differences: ${differencesCorrect ? 'PASS' : 'FAIL'}\n`);

// Summary
console.log('=== TEST SUMMARY ===');
console.log('====================');

const tests = [
  { name: 'Position count (331)', pass: hexPositions.length === 331 },
  { name: 'Cube coordinates (x+y+z=0)', pass: coordinateViolations === 0 },
  { name: 'Line axes count (3)', pass: lineDirections.length === 3 },
  { name: 'Neighbor count (6)', pass: neighbors.length === 6 },
  { name: 'Adjacency types', pass: allHexagonal },
  { name: 'Distance calculation', pass: distance === expectedDistance },
  { name: 'Edge detection', pass: incorrectEdges === 0 },
  { name: 'Board differences', pass: differencesCorrect },
];

const passedTests = tests.filter((t) => t.pass).length;
const totalTests = tests.length;

tests.forEach((test) => {
  const status = test.pass ? '✓ PASS' : '✗ FAIL';
  console.log(`${status}: ${test.name}`);
});

console.log(`\nTotal: ${passedTests}/${totalTests} tests passed`);

if (passedTests === totalTests) {
  console.log('\n🎉 ALL TESTS PASSED! Hexagonal board implementation is correct.');
  process.exit(0);
} else {
  console.log(
    `\n❌ ${totalTests - passedTests} test(s) failed. Please review hexagonal board implementation.`
  );
  process.exit(1);
}
